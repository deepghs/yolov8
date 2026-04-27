import datetime
import os
from functools import partial
from tempfile import TemporaryDirectory
from typing import Optional

import click
from ditk import logging
from huggingface_hub import HfApi, CommitOperationAdd

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
@click.option('--with-embedding/--no-embedding', 'with_embedding', default=False,
              show_default=True,
              help='Also publish model_with_embedding.onnx alongside '
                   'model.onnx. The dual-head ONNX has an additional '
                   '"embedding" output for retrieval / dedup / FAISS use; '
                   'consumers that only need the detection head can keep '
                   'using model.onnx (output names are unchanged).')
@click.option('--dry-run', 'dry_run', is_flag=True, default=False,
              help='Run the export pipeline and report the upload manifest, '
                   'but do not create the HF repo or commit. HF_TOKEN is not '
                   'required in this mode.')
def huggingface(workdir: str, name: Optional[str],
                repository: str, revision: str, opset_version: int = 14,
                with_embedding: bool = False,
                dry_run: bool = False):
    """Bundle a workdir's best checkpoint and upload it to a HF model repo.

    Materialises the same export bundle that
    :func:`yolov8.export.export_model_from_workdir` produces (anonymised
    ``best.pt``, ``model.onnx`` with full ult / dghs metadata + an
    embedded ``threshold.json``, plot/metrics PNGs, ...) into a
    temporary directory, then commits everything under
    ``<repository>/<name>/`` on the chosen branch. ``--with-embedding``
    additionally publishes ``model_with_embedding.onnx`` from
    :func:`yolov8.onnx.export_yolo_to_onnx_with_embedding`.

    :param workdir: Source training directory.
    :type workdir: str
    :param name: Display name. Defaults to the workdir's basename.
    :type name: str or None
    :param repository: HF model repo, e.g. ``deepghs/anime_object_yolo``.
    :type repository: str
    :param revision: Branch / tag / SHA to commit onto.
    :type revision: str
    :param opset_version: ONNX opset for the exported graph.
    :type opset_version: int
    :param with_embedding: Also publish a dual-head ONNX with an
        ``embedding`` output for retrieval / dedup / FAISS.
    :type with_embedding: bool
    :param dry_run: Report the upload manifest but skip repo
        creation and the actual commit. ``HF_TOKEN`` is not needed
        in this mode.
    :type dry_run: bool

    Example::

        >>> # CLI form:
        >>> #   HF_TOKEN=... python -m yolov8.publish huggingface \\
        >>> #       -w runs/my_train -r deepghs/foo --with-embedding
        >>> from yolov8.publish import huggingface  # the click callback
    """
    logging.try_init_root(logging.INFO)

    if dry_run:
        hf_client = None
    else:
        hf_client = HfApi(token=os.environ['HF_TOKEN'])
        logging.info(f'Initialize repository {repository!r}')
        hf_client.create_repo(repo_id=repository, repo_type='model', exist_ok=True)

    with TemporaryDirectory() as td:
        name = name or os.path.basename(os.path.abspath(workdir))
        # threshold.json is now also embedded inside model.onnx's
        # metadata_props (under ``dghs.yolov8.threshold``) by
        # export_yolo_to_onnx, so the on-HF artifact is self-describing.
        # We still ship the sidecar threshold.json for backwards
        # compatibility with HF readers that expected it as a top-level
        # file - export_model_from_workdir handles that.
        files = export_model_from_workdir(workdir, td, name,
                                           opset_version=opset_version,
                                           with_embedding=with_embedding)

        if dry_run:
            total = 0
            logging.info(f'[dry-run] would publish {len(files)} files to '
                         f'{repository!r} on revision {revision!r}:')
            for local_file, filename in files:
                size = os.path.getsize(local_file)
                total += size
                logging.info(f'[dry-run]   {name}/{filename}  ({size:,} bytes)')
            logging.info(f'[dry-run] total: {total:,} bytes ({total / 1e6:.2f} MB)')
            return

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
    """Deploy a workdir to a Roboflow project version (legacy path).

    Optional integration: requires the ``roboflow`` package, which is
    not in the main requirements. Install via
    ``pip install -r requirements-roboflow.txt`` (this also downgrades
    ultralytics to 8.0.196 for Roboflow SDK compatibility).

    :param workdir: Source training directory.
    :type workdir: str
    :param project: Roboflow project path of the form
        ``<workspace>/<project>``.
    :type project: str
    :param version: Project version number to deploy to.
    :type version: int
    :raises click.ClickException: If the optional ``roboflow``
        package is not installed.

    Example::

        >>> # CLI form:
        >>> #   ROBOFLOW_APIKEY=... python -m yolov8.publish roboflow \\
        >>> #       -w runs/my_train -p ws/proj -v 3
        >>> from yolov8.publish import roboflow  # click callback
    """
    logging.try_init_root(logging.INFO)

    try:
        from roboflow import Roboflow
    except ImportError as err:
        raise click.ClickException(
            "Roboflow integration is an optional extra and is not installed. "
            "Install it with `pip install -r requirements-roboflow.txt` from "
            "the repo root. Note that this will downgrade ultralytics to "
            "8.0.196 for Roboflow SDK compatibility."
        ) from err

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
