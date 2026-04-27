"""CLI helpers shared across the click entry points."""
import click
from click.core import Context, Option

#: Default click context: enables ``-h`` as an alias for ``--help`` for
#: every command in the package.
GLOBAL_CONTEXT_SETTINGS = dict(
    help_option_names=['-h', '--help']
)


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
