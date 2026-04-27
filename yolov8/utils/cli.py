"""CLI helpers shared across the click entry points."""
import json
import re
from typing import Any, Sequence, Tuple, Union

import click
from click.core import Context, Option, Parameter

#: Default click context: enables ``-h`` as an alias for ``--help`` for
#: every command in the package.
GLOBAL_CONTEXT_SETTINGS = dict(
    help_option_names=['-h', '--help']
)


# ---------------------------------------------------------------------------
# Hyperparameter parsing for ``-p key=value`` style CLI flags
# ---------------------------------------------------------------------------

def parse_hyperparam(token: str) -> Tuple[str, Any]:
    """Parse a single ``key=value`` token into a typed Python value.

    The value is decoded with :func:`json.loads` first, which yields the
    natural Python form for ``int`` / ``float`` / ``bool`` / ``null`` /
    list / dict literals. If JSON decoding fails, the value is returned
    as a plain string. This covers everything Ultralytics' trainer
    accepts on its keyword interface:

    * ``batch=16``               -> ``16`` (int)
    * ``mosaic=1.0``             -> ``1.0`` (float)
    * ``cos_lr=true``            -> ``True`` (bool)
    * ``device=null``            -> ``None``
    * ``device=[0,1,2,3]``       -> ``[0, 1, 2, 3]`` (list)
    * ``optimizer=AdamW``        -> ``"AdamW"`` (string)
    * ``name=my-run``            -> ``"my-run"`` (string)

    :param token: A single ``key=value`` string. Whitespace around the
        ``key`` and the ``value`` is stripped.
    :type token: str
    :returns: ``(key, decoded_value)``.
    :rtype: tuple[str, Any]
    :raises click.BadParameter: When the token has no ``=`` separator or
        the key is empty.

    Example::

        >>> from yolov8.utils import parse_hyperparam
        >>> parse_hyperparam("batch=16")
        ('batch', 16)
        >>> parse_hyperparam("device=[0,1]")
        ('device', [0, 1])
        >>> parse_hyperparam("optimizer=AdamW")
        ('optimizer', 'AdamW')
    """
    if "=" not in token:
        raise click.BadParameter(
            f"hyperparameter must look like KEY=VALUE; got {token!r}"
        )
    key, _, value = token.partition("=")
    key = key.strip()
    value = value.strip()
    if not key:
        raise click.BadParameter(f"empty key in {token!r}")
    if not _VALID_KEY_RE.fullmatch(key):
        raise click.BadParameter(
            f"hyperparameter key {key!r} is not a valid Python identifier "
            f"(letters, digits, underscores; cannot start with a digit)"
        )
    try:
        decoded: Any = json.loads(value)
    except json.JSONDecodeError:
        decoded = value
    return key, decoded


_VALID_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def parse_hyperparams(tokens: Sequence[str]) -> dict:
    """Parse a sequence of ``key=value`` tokens into a dict.

    Later occurrences override earlier ones for the same key, mirroring
    the Ultralytics CLI / argparse semantics.

    :param tokens: The raw strings collected from ``-p`` repetitions.
    :type tokens: Sequence[str]
    :returns: ``{key: decoded_value}`` for every parsed token.
    :rtype: dict
    :raises click.BadParameter: Propagated from
        :func:`parse_hyperparam` when any token is malformed.

    Example::

        >>> from yolov8.utils import parse_hyperparams
        >>> parse_hyperparams(["batch=16", "imgsz=640", "device=[0,1]"])
        {'batch': 16, 'imgsz': 640, 'device': [0, 1]}
    """
    out: dict = {}
    for tok in tokens:
        k, v = parse_hyperparam(tok)
        out[k] = v
    return out


def hyperparam_callback(ctx: Context, param: Parameter,
                        value: Sequence[str]) -> dict:
    """Click ``callback`` for a repeatable ``-p key=value`` option.

    Wired up via::

        @click.option("-p", "--hyperparam", "hyperparams",
                      multiple=True, callback=hyperparam_callback,
                      ...)

    The collected tuple is fed into :func:`parse_hyperparams` and the
    resulting dict replaces the raw tuple in the command's keyword
    arguments. Click surfaces any :class:`click.BadParameter` raised
    here as a normal CLI usage error.

    :param ctx: Active click context. Unused; accepted for API shape.
    :type ctx: click.core.Context
    :param param: The option this callback is attached to. Unused;
        accepted for API shape.
    :type param: click.core.Parameter
    :param value: The tuple of raw strings click collected for the
        repeatable option.
    :type value: Sequence[str]
    :returns: Parsed ``{key: value}`` dict (empty if the user did not
        pass any ``-p``).
    :rtype: dict

    Example::

        >>> # In a click command:
        >>> # @click.option("-p", "--hyperparam", "hyperparams",
        >>> #              multiple=True, callback=hyperparam_callback)
        >>> # def cmd(hyperparams): ...
    """
    _ = ctx, param  # accepted to match the click callback signature
    if not value:
        return {}
    return parse_hyperparams(value)


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
