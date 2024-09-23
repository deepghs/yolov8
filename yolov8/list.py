import os.path

import click
import pandas as pd
from ditk import logging
from hfutils.operate import get_hf_fs, get_hf_client
from hfutils.utils import hf_fs_path
from huggingface_hub.hf_api import RepoFile
from tqdm import tqdm
from ultralytics import YOLO
from ultralytics.utils.torch_utils import get_flops_with_torch_profiler, get_num_params

from yolov8.utils import GLOBAL_CONTEXT_SETTINGS


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

    for pt_file in tqdm(hf_fs.glob(hf_fs_path(
            repo_id=repository,
            repo_type='model',
            filename='*/model.pt',
            revision=revision,
    ))):
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
        rows.append({
            'Model': name,
            'FLOPS': get_flops_with_torch_profiler(model),
            'Params': get_num_params(model.model),
            **(model.ckpt.get('train_metrics') or {}),
            'Labels': ', '.join(map(lambda x: f'`{x}`', labels)),
            'created_at': last_commit_at,
        })

    df = pd.DataFrame(rows)
    df = df.sort_values(by=['created_at'], ascending=[False])
    del df['created_at']
    print(df.to_markdown(index=False, numalign="center", stralign="center"))


if __name__ == '__main__':
    pass
