import io
import os.path

import click
import numpy as np
import pandas as pd
from ditk import logging
from hbutils.system import TemporaryDirectory
from hfutils.operate import get_hf_fs, get_hf_client, upload_directory_as_directory
from hfutils.utils import hf_fs_path, parse_hf_fs_path
from huggingface_hub.hf_api import RepoFile
from tqdm import tqdm
from ultralytics import YOLO
from ultralytics.utils.torch_utils import get_flops_with_torch_profiler, get_num_params

from .utils import GLOBAL_CONTEXT_SETTINGS, float_pe, markdown_to_df


@click.command('huggingface', context_settings={**GLOBAL_CONTEXT_SETTINGS},
               help='Publish model to huggingface model repository')
@click.option('--repository', '-r', 'repository', type=str, required=True,
              help='Repository for publishing model.', show_default=True)
@click.option('--revision', '-R', 'revision', type=str, default='main',
              help='Revision for pushing the model.', show_default=True)
def list_(repository: str, revision: str = 'main'):
    logging.try_init_root(logging.INFO)
    rows = []
    hf_fs = get_hf_fs()
    hf_client = get_hf_client()

    for pt_path in tqdm(hf_fs.glob(hf_fs_path(
            repo_id=repository,
            repo_type='model',
            filename='*/model.pt',
            revision=revision,
    ))):
        pt_file = parse_hf_fs_path(pt_path).filename
        name = os.path.dirname(pt_file)
        logging.info(f'Making information for {name!r} ...')
        model = YOLO(hf_client.hf_hub_download(
            repo_id=repository,
            repo_type='model',
            filename=pt_file,
            revision=revision,
        ))

        repo_file: RepoFile = list(hf_client.get_paths_info(
            repo_id=repository,
            repo_type='model',
            paths=[f'{name}/model.pt'],
            expand=True,
        ))[0]
        last_commit_at = repo_file.last_commit.date.timestamp()

        names_map = model.names
        labels = [names_map[i] for i in range(len(names_map))]
        metrics = {
            key.split('/', maxsplit=1)[-1]: value
            for key, value in dict(model.ckpt.get('train_metrics') or {}).items()
            if key.startswith('metrics/')
        }
        rows.append({
            'Model': name,
            'FLOPS': float_pe(get_flops_with_torch_profiler(model)),
            'Params': float_pe(get_num_params(model.model)),
            **metrics,
            'Labels': ', '.join(map(lambda x: f'`{x}`', labels)),
            'created_at': last_commit_at,
        })

    df = pd.DataFrame(rows)
    df = df.sort_values(by=['created_at'], ascending=[False])
    del df['created_at']
    df = df.replace(np.nan, 'N/A')

    with TemporaryDirectory() as td:
        with open(os.path.join(td, 'README.md'), 'w') as f:
            if not hf_fs.exists(hf_fs_path(
                    repo_id=repository,
                    repo_type='model',
                    filename='README.md',
                    revision=revision,
            )):
                print(df.to_markdown(index=False, numalign="center", stralign="center"), file=f)

            else:
                table_printed = False
                tb_lines = []
                with io.StringIO(hf_fs.read_text(hf_fs_path(
                        repo_id=repository,
                        repo_type='model',
                        filename='README.md',
                        revision=revision,
                )).rstrip() + os.linesep * 2) as ifx:
                    for line in ifx:
                        line = line.rstrip()
                        if line.startswith('|') and not table_printed:
                            tb_lines.append(line)
                        else:
                            if tb_lines:
                                df = markdown_to_df(os.linesep.join(tb_lines))
                                if 'Model' in df.columns and 'FLOPS' in df.columns and \
                                        'Params' in df.columns and 'Labels' in df.columns:
                                    print(df.to_markdown(index=False, numalign="center", stralign="center"), file=f)
                                    table_printed = True
                                    tb_lines.clear()
                            print(line, file=f)

                if not table_printed:
                    print(df.to_markdown(index=False, numalign="center", stralign="center"), file=f)

        upload_directory_as_directory(
            repo_id=repository,
            repo_type='model',
            revision=revision,
            path_in_repo='.',
            local_directory=td,
            message=f'Sync README for {repository}',
            hf_token=os.environ.get('HF_TOKEN'),
        )


if __name__ == '__main__':
    list_()
