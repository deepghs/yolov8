import datetime
import os
from functools import partial
from tempfile import TemporaryDirectory
from typing import Optional

import click
from ditk import logging
from huggingface_hub import HfApi, CommitOperationAdd
from roboflow import Roboflow

from .export import export_model_from_workdir
from .utils import GLOBAL_CONTEXT_SETTINGS
from .utils import print_version as _origin_print_version

print_version = partial(_origin_print_version, 'publish')


@click.group(context_settings={**GLOBAL_CONTEXT_SETTINGS})
@click.option('-v', '--version', is_flag=True,
              callback=print_version, expose_value=False, is_eager=True,
              help="Utils publishing models.")
def cli():
    pass  # pragma: no cover


@cli.command('huggingface', context_settings={**GLOBAL_CONTEXT_SETTINGS},
             help='Publish model to huggingface model repository')
@click.option('--workdir', '-w', 'workdir', type=click.Path(file_okay=False, exists=True), required=True,
              help='Work directory of the training.', show_default=True)
@click.option('--name', '-n', 'name', type=str, default=None,
              help='Name of the checkpoint. Default is the basename of the work directory.', show_default=True)
@click.option('--repository', '-r', 'repository', type=str, required=True,
              help='Repository for publishing model.', show_default=True)
@click.option('--revision', '-R', 'revision', type=str, default='main',
              help='Revision for pushing the model.', show_default=True)
@click.option('--opset_version', 'opset_version', type=int, default=14,
              help='Version of OP set.', show_default=True)
def huggingface(workdir: str, name: Optional[str],
                repository: str, revision: str, opset_version: int = 14):
    logging.try_init_root(logging.INFO)

    hf_client = HfApi(token=os.environ['HF_TOKEN'])
    logging.info(f'Initialize repository {repository!r}')
    hf_client.create_repo(repo_id=repository, repo_type='model', exist_ok=True)

    with TemporaryDirectory() as td:
        name = name or os.path.basename(os.path.abspath(workdir))
        files = export_model_from_workdir(workdir, td, name, opset_version=opset_version)
        current_time = datetime.datetime.now().astimezone().strftime('%Y-%m-%d %H:%M:%S %Z')
        commit_message = f"Publish model {name}, on {current_time}"
        logging.info(f'Publishing model {name!r} to repository {repository!r} ...')
        hf_client.create_commit(
            repository,
            [
                CommitOperationAdd(
                    path_in_repo=f'{name}/{filename}',
                    path_or_fileobj=local_file,
                ) for local_file, filename in files
            ],
            commit_message=commit_message,
            repo_type='model',
            revision=revision,
        )


@cli.command('roboflow', context_settings={**GLOBAL_CONTEXT_SETTINGS},
             help='Publish model to huggingface model repository')
@click.option('--workdir', '-w', 'workdir', type=click.Path(file_okay=False, exists=True), required=True,
              help='Work directory of the training.', show_default=True)
@click.option('--project', '-p', 'project', type=str, required=True,
              help='Project to deploy to, e.g. anime-cv/anime_person_detection', show_default=True)
@click.option('--version', '-v', 'version', type=int, required=True,
              help='Version in project.', show_default=True)
def roboflow(workdir: str, project: str, version: int):
    logging.try_init_root(logging.INFO)

    rf = Roboflow(api_key=os.environ['ROBOFLOW_APIKEY'])
    workspace, project_name = project.split('/', maxsplit=2)
    proj = rf.workspace(workspace).project(project_name)
    version = proj.version(version)
    logging.info(f'Meta-information of version {version!r} in project {project!r}:\n{version}')

    if not workdir.endswith('/'):
        workdir = f'{workdir}/'
    logging.info(f'Deploying {workdir!r} as yolov8 ...')
    version.deploy("yolov8", workdir)


if __name__ == '__main__':
    cli()
