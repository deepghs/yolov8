"""Entry point for ``python -m yolov8.train``.

Wires the click command group from :mod:`yolov8.train._cli` into the
``__main__`` slot so the CLI is reachable as a module-level command.

Run::

    python -m yolov8.train --help
    python -m yolov8.train detect -d coco8.yaml
    python -m yolov8.train segment -d /path/to/data.yaml
"""
from ._cli import cli


if __name__ == '__main__':
    cli()
