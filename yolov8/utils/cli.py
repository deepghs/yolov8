"""Click-specific helpers shared across the package's CLIs.

Most of the heavy lifting (typed-schema parsing of ``-p key=value``
flags) lives in :mod:`yolov8.utils.parser` so that non-CLI callers
can use the parser without dragging click in. The same parser symbols
are re-exported from this module for backwards compatibility — this
module is the historical home of the public API.
"""
from __future__ import annotations

import re
from typing import Union

import click
from click.core import Context, Option

# Re-export the parser surface so existing imports
# ``from yolov8.utils.cli import parse_hyperparam`` keep working.
from .parser import (  # noqa: F401
    ParseError,
    _UNION_ORIGINS,
    _auto_detect,
    _parse_bool,
    fraction,
    hyperparam_callback,
    hyperparam_callback_factory,
    parse_hyperparam,
    parse_hyperparams,
    parse_value,
    yolo_train_param_schema,
)

#: Default click context: enables ``-h`` as an alias for ``--help``
#: across every command in the package.
GLOBAL_CONTEXT_SETTINGS = dict(
    help_option_names=['-h', '--help']
)


def parse_yversion(yversion: Union[str, int]) -> Union[str, int]:
    """Normalise a CLI ``yversion`` argument the way the original
    ``test_train_*.py`` recipes did.

    Strings of pure digits are converted to ``int`` (so the trainer
    treats them as YOLO family numbers); the special ``"rtdetr"`` /
    ``"world"`` strings stay as-is; anything else passes through
    unchanged.

    :param yversion: CLI input. Accepted shapes: ``"8"`` / ``"11"`` /
        ``11`` / ``"rtdetr"``.
    :type yversion: str or int
    :returns: ``int`` for numeric versions, ``str`` for named ones.
    :rtype: int or str

    Example::

        >>> from yolov8.utils import parse_yversion
        >>> parse_yversion("11"), parse_yversion("rtdetr"), parse_yversion(8)
        (11, 'rtdetr', 8)
    """
    if isinstance(yversion, int):
        return yversion
    s = str(yversion).strip()
    if re.fullmatch(r"^\d+$", s):
        return int(s)
    return s


def print_version(module: str, ctx: Context, param: Option, value: bool) -> None:
    """Print the calling CLI's version banner and exit.

    Wired up via ``click.option(..., callback=print_version,
    is_eager=True)`` on each ``-v/--version`` flag so the banner is
    emitted before any other option is resolved.

    :param module: Display name of the calling module (e.g.
        ``"export"`` or ``"publish"``). Embedded into the banner.
    :type module: str
    :param ctx: Active click context. Used to short-circuit the
        command after printing.
    :type ctx: click.core.Context
    :param param: The click option whose callback this is. Unused;
        accepted to match the click callback signature.
    :type param: click.core.Option
    :param value: ``True`` when the flag was passed on the command
        line. ``False`` (or ``None`` during resilient parsing) is a
        no-op.
    :type value: bool

    Example::

        >>> from functools import partial
        >>> from yolov8.utils import print_version
        >>> _print = partial(print_version, "demo")  # used as click callback
    """
    _ = param
    if not value or ctx.resilient_parsing:
        return  # pragma: no cover

    click.echo(f'Module utils of {module}')
    ctx.exit()
