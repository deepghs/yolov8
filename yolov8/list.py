import click

from yolov8.utils import GLOBAL_CONTEXT_SETTINGS


@click.command('huggingface', context_settings={**GLOBAL_CONTEXT_SETTINGS},
               help='Publish model to huggingface model repository')
@click.option('--repository', '-r', 'repository', type=str, required=True,
              help='Repository for publishing model.', show_default=True)
@click.option('--revision', '-R', 'revision', type=str, default='main',
              help='Revision for pushing the model.', show_default=True)
def list_(repository: str, revision: str = 'main'):
    pass


if __name__ == '__main__':
    pass
