"""CLI helpers shared across the click entry points.

The core piece is :func:`parse_hyperparam` and its sequence form
:func:`parse_hyperparams`. Both accept an optional ``schema`` dict that
maps each parameter name to a Python ``type`` / :mod:`typing` generic /
callable; values are coerced and validated against that spec. Without
a schema entry the parser falls back to auto-detect (JSON-style
literals → typed Python value, otherwise a plain string).

Schema vocabulary:

* ``int`` / ``float`` / ``bool`` / ``str`` — plain Python types. ``str``
  short-circuits any auto-detection: the raw token survives verbatim,
  so ``parse_hyperparam('at=111', {'at': str})`` returns
  ``('at', '111')``.
* ``list[T]`` — comma-separated input (``1,2,3`` style — no brackets).
  Each element is sub-parsed by ``T``.
* ``typing.Union[A, B, ...]`` — try each type in order, return the
  first that succeeds.
* any callable ``f(raw: str) -> Any`` — custom coercion. Use this for
  range-checked floats, choice validation, etc. :func:`fraction` ships
  one for ``[0.0, 1.0]`` floats.

A :func:`yolo_train_param_schema` factory builds the schema for
Ultralytics ``model.train(...)`` keywords by reading the upstream
``CFG_INT_KEYS`` / ``CFG_FLOAT_KEYS`` / ``CFG_FRACTION_KEYS`` /
``CFG_BOOL_KEYS`` constants, plus a few special cases (``device``,
``imgsz``, ``name`` / ``model`` / ``data`` / ``project``, ``classes``,
``freeze``).
"""
from __future__ import annotations

import json
import re
import sys
import typing
from typing import Any, Callable, List, Mapping, Optional, Sequence, Tuple, Union

import click
from click.core import Context, Option, Parameter

# PEP 604 (Python 3.10+): ``X | Y`` evaluates to ``types.UnionType`` at
# runtime, distinct from the legacy ``typing.Union`` returned by
# ``Union[X, Y]``. ``typing.get_origin`` returns the corresponding form
# for each. We treat them as equivalent so user-supplied schemas can
# spell unions either way without surprising the parser. Conditional
# import keeps the module loadable on Python 3.8 / 3.9 where
# ``types.UnionType`` does not exist yet.
if sys.version_info >= (3, 10):
    import types as _types

    _UNION_ORIGINS: Tuple[Any, ...] = (Union, _types.UnionType)
else:  # pragma: no cover - exercised on 3.8/3.9 envs
    _UNION_ORIGINS = (Union,)

#: Default click context: enables ``-h`` as an alias for ``--help`` for
#: every command in the package.
GLOBAL_CONTEXT_SETTINGS = dict(
    help_option_names=['-h', '--help']
)


# ---------------------------------------------------------------------------
# Hyperparameter parsing for ``-p key=value`` style CLI flags
# ---------------------------------------------------------------------------

_VALID_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_BOOL_TRUE = {"true", "1", "yes", "on"}
_BOOL_FALSE = {"false", "0", "no", "off"}
_NULL_LITERALS = {"null", "none", ""}


class ParseError(ValueError):
    """Raised by :func:`parse_hyperparam` when a value does not match
    its schema entry. Wrapped into :class:`click.BadParameter` by the
    higher-level callback so users get a clean CLI error."""


def fraction(raw: str) -> float:
    """Coerce ``raw`` to ``float`` and require ``0.0 <= value <= 1.0``.

    Designed for use as a schema entry, mirroring Ultralytics'
    ``CFG_FRACTION_KEYS`` validation::

        {'mosaic': fraction, 'mixup': fraction, ...}

    :param raw: Raw CLI value.
    :type raw: str
    :returns: Parsed float in ``[0.0, 1.0]``.
    :rtype: float
    :raises ParseError: If the value cannot be parsed as float or is
        outside ``[0.0, 1.0]``.

    Example::

        >>> from yolov8.utils.cli import fraction
        >>> fraction("0.5")
        0.5
        >>> try:
        ...     fraction("2.0")
        ... except Exception as e:
        ...     print(type(e).__name__)
        ParseError
    """
    try:
        v = float(raw)
    except (TypeError, ValueError) as err:
        raise ParseError(f"cannot parse {raw!r} as float") from err
    if not 0.0 <= v <= 1.0:
        raise ParseError(f"value {v} out of range [0.0, 1.0]")
    return v


def _parse_bool(raw: str) -> bool:
    """Coerce a raw string to ``bool`` accepting common truthy/falsy
    spellings (case-insensitive). ``true / false / 1 / 0 / yes / no /
    on / off`` all work."""
    s = raw.strip().lower()
    if s in _BOOL_TRUE:
        return True
    if s in _BOOL_FALSE:
        return False
    raise ParseError(
        f"cannot interpret {raw!r} as bool; "
        f"accepted: true/false/1/0/yes/no/on/off"
    )


def _type_name(spec: Any) -> str:
    """Best-effort human-readable name for a schema entry, used in
    error messages."""
    if spec is None:
        return "auto"
    if isinstance(spec, type):
        return spec.__name__
    origin = typing.get_origin(spec)
    if origin is list:
        args = typing.get_args(spec)
        return f"list[{_type_name(args[0]) if args else 'any'}]"
    if origin in _UNION_ORIGINS:
        return " | ".join(_type_name(a) for a in typing.get_args(spec))
    if callable(spec):
        return getattr(spec, "__name__", repr(spec))
    return repr(spec)


def _coerce_typed(raw: str, spec: Any) -> Any:
    """Coerce a raw string per a schema entry (a Python ``type``, a
    :mod:`typing` generic, or a callable).

    Recursive: ``list[T]`` walks each comma-separated element through
    :func:`_coerce_typed` against ``T``; ``Union[...]`` tries each
    member in order until one succeeds.
    """
    if spec is None:
        return _auto_detect(raw)

    # Atomic Python types.
    if spec is str:
        return raw
    if spec is bool:
        return _parse_bool(raw)
    if spec is int:
        try:
            return int(raw)
        except (TypeError, ValueError) as err:
            raise ParseError(f"cannot parse {raw!r} as int") from err
    if spec is float:
        try:
            return float(raw)
        except (TypeError, ValueError) as err:
            raise ParseError(f"cannot parse {raw!r} as float") from err
    if spec is type(None):
        if raw.strip().lower() in _NULL_LITERALS:
            return None
        raise ParseError(f"only 'null'/'none' parses as None, got {raw!r}")

    # Generic typing constructs.
    origin = typing.get_origin(spec)
    if origin is list:
        item_args = typing.get_args(spec)
        item_spec = item_args[0] if item_args else None
        # ``-p k=`` (empty value) is a 0-element list, not [None].
        if raw.strip() == "":
            return []
        items = [s.strip() for s in raw.split(",")]
        return [_coerce_typed(s, item_spec) for s in items]
    if origin in _UNION_ORIGINS:
        last_err: Optional[Exception] = None
        for member in typing.get_args(spec):
            # Disambiguation: when a Union member is ``list[T]`` but
            # the input has no comma, skip it. This makes
            # ``Union[int, str, list[int]]`` behave intuitively:
            # ``cpu`` -> 'cpu' (str), ``0`` -> 0 (int), ``0,1,2,3`` ->
            # [0, 1, 2, 3] (list). Without this, the bare ``str``
            # branch always wins and ``0,1,2,3`` would become the
            # literal string ``'0,1,2,3'``.
            if typing.get_origin(member) is list and "," not in raw:
                continue
            try:
                return _coerce_typed(raw, member)
            except (ParseError, ValueError, TypeError) as err:
                last_err = err
        # All branches failed; surface the most informative error.
        raise ParseError(
            f"value {raw!r} matches none of {_type_name(spec)}"
        ) from last_err

    # Custom callable parser.
    if callable(spec):
        try:
            return spec(raw)
        except (ParseError, ValueError, TypeError) as err:
            raise ParseError(
                f"value {raw!r} rejected by {_type_name(spec)}: {err}"
            ) from err

    raise TypeError(f"unsupported schema type: {spec!r}")


def _auto_detect(raw: str) -> Any:
    """No-schema fallback: try ``json.loads`` first, fall back to the
    raw string. ``json`` covers int / float / bool / null / list / dict
    literals, exactly the cases :class:`ultralytics.cfg.IterableSimpleNamespace`
    expects on its keyword interface."""
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


def parse_hyperparam(token: str,
                     schema: Optional[Mapping[str, Any]] = None
                     ) -> Tuple[str, Any]:
    """Parse a single ``key=value`` token into a typed Python value.

    Behaviour summary:

    * ``schema is None`` (or the key isn't in it): auto-detect via
      :func:`json.loads`; fall back to a plain ``str``.
    * Schema entry ``str``: the raw token survives verbatim — useful
      for arguments like ``name=2024-04-27`` where you don't want
      ``2024`` to become an int.
    * Schema entry ``int`` / ``float`` / ``bool`` / callable: strict
      coercion. Errors surface as :class:`click.BadParameter`.
    * Schema entry ``list[T]``: comma-separated input, e.g.
      ``device=0,1,2,3``. Each element goes through ``_coerce_typed(T)``.
    * Schema entry ``typing.Union[A, B, ...]``: try in order.

    :param token: ``key=value`` string. Whitespace around either side
        of ``=`` is stripped.
    :type token: str
    :param schema: Optional ``{key: spec}`` mapping. ``None`` to fall
        back to auto-detect for every key.
    :type schema: Mapping[str, Any] or None
    :returns: ``(key, decoded_value)``.
    :rtype: tuple[str, Any]
    :raises click.BadParameter: When the token is malformed or the
        coercion fails against the schema entry.

    Example::

        >>> from yolov8.utils.cli import parse_hyperparam
        >>> parse_hyperparam("batch=16")
        ('batch', 16)
        >>> parse_hyperparam("device=0,1,2,3", {"device": list[int]})
        ('device', [0, 1, 2, 3])
        >>> parse_hyperparam("at=111", {"at": str})
        ('at', '111')
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

    spec = schema.get(key) if schema is not None else None
    if spec is None:
        return key, _auto_detect(value)

    try:
        decoded = _coerce_typed(value, spec)
    except ParseError as err:
        raise click.BadParameter(
            f"hyperparameter {key!r} expects {_type_name(spec)}: {err}"
        ) from err
    return key, decoded


def parse_hyperparams(tokens: Sequence[str],
                      schema: Optional[Mapping[str, Any]] = None
                      ) -> dict:
    """Parse a sequence of ``key=value`` tokens into a dict.

    Later occurrences override earlier ones for the same key,
    mirroring Ultralytics / argparse semantics.

    :param tokens: Raw strings collected from a repeatable ``-p``
        click option.
    :type tokens: Sequence[str]
    :param schema: Optional schema, see :func:`parse_hyperparam`.
    :type schema: Mapping[str, Any] or None
    :returns: ``{key: decoded_value}`` for every parsed token.
    :rtype: dict
    :raises click.BadParameter: Propagated from
        :func:`parse_hyperparam` for any malformed token.

    Example::

        >>> from yolov8.utils.cli import parse_hyperparams
        >>> schema = {"batch": int, "device": list[int]}
        >>> parse_hyperparams(["batch=16", "device=0,1"], schema)
        {'batch': 16, 'device': [0, 1]}
    """
    out: dict = {}
    for tok in tokens:
        k, v = parse_hyperparam(tok, schema=schema)
        out[k] = v
    return out


def hyperparam_callback(ctx: Context, param: Parameter,
                        value: Sequence[str]) -> dict:
    """Click ``callback`` for a repeatable ``-p key=value`` option,
    schema-less variant.

    Use :func:`hyperparam_callback_factory` when you need schema-aware
    coercion (the typical case for production CLIs).

    :param ctx: Active click context. Unused; accepted for API shape.
    :type ctx: click.core.Context
    :param param: The option this callback is attached to. Unused.
    :type param: click.core.Parameter
    :param value: Tuple of raw strings click collected.
    :type value: Sequence[str]
    :returns: ``{key: value}`` parsed by :func:`parse_hyperparams` with
        no schema (auto-detect everywhere).
    :rtype: dict
    """
    _ = ctx, param
    if not value:
        return {}
    return parse_hyperparams(value)


def hyperparam_callback_factory(
    schema: Optional[Mapping[str, Any]],
) -> Callable[[Context, Parameter, Sequence[str]], dict]:
    """Build a click callback bound to a specific schema.

    Use as::

        cb = hyperparam_callback_factory(my_schema)

        @click.option('-p', '--hyperparam', 'hyperparams',
                      multiple=True, callback=cb, ...)

    The factory keeps the schema in the closure so the same
    :func:`parse_hyperparams` plumbing can be reused for multiple CLIs
    with different keyword vocabularies.

    :param schema: Schema mapping, see :func:`parse_hyperparam`.
        ``None`` is permitted and disables coercion (equivalent to
        :func:`hyperparam_callback`).
    :type schema: Mapping[str, Any] or None
    :returns: A click callback function.
    :rtype: Callable[[Context, Parameter, Sequence[str]], dict]

    Example::

        >>> from yolov8.utils.cli import hyperparam_callback_factory
        >>> cb = hyperparam_callback_factory({"batch": int})
        >>> # In a click command:
        >>> # @click.option('-p', multiple=True, callback=cb)
        >>> # def cmd(p): ...
    """

    def _cb(ctx: Context, param: Parameter,
            value: Sequence[str]) -> dict:
        _ = ctx, param
        if not value:
            return {}
        return parse_hyperparams(value, schema=schema)

    return _cb


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


# ---------------------------------------------------------------------------
# YOLO train schema, derived from ultralytics' own typing constants
# ---------------------------------------------------------------------------

def yolo_train_param_schema() -> dict:
    """Build the ``parse_hyperparams`` schema for Ultralytics
    ``model.train(...)`` keywords.

    Reads the upstream typing constants
    (``ultralytics.cfg.{CFG_INT_KEYS, CFG_FLOAT_KEYS,
    CFG_FRACTION_KEYS, CFG_BOOL_KEYS}``) and translates them into the
    schema vocabulary this module accepts:

    * ``CFG_INT_KEYS``       -> ``int``
    * ``CFG_FLOAT_KEYS``     -> ``Union[int, float]`` (Ultralytics'
                                  own ``check_cfg`` accepts both for
                                  these keys)
    * ``CFG_FRACTION_KEYS``  -> :func:`fraction`  (range-checked
                                  ``[0.0, 1.0]``)
    * ``CFG_BOOL_KEYS``      -> ``bool``

    Plus a handful of special cases that the plain CFG sets don't
    cover:

    * ``device``    accepts ``int`` / ``str`` / ``list[Union[int, str]]``
                    so ``-p device=0,1,2,3`` and ``-p device=cpu`` both
                    work.
    * ``imgsz``     accepts ``int`` or ``list[int]`` for non-square
                    train inputs (``imgsz=640,480``).
    * ``classes``   ``list[int]`` for the validator's class filter.
    * ``freeze``    ``int`` or ``list[int]`` for backbone freezing.
    * ``name`` / ``project`` / ``model`` / ``data`` are forced to
                    ``str`` so date / git-sha-like values aren't
                    auto-coerced.

    Builds the dict at call time so updates to Ultralytics flow
    through transparently.

    :returns: Schema dict suitable for :func:`parse_hyperparams`.
    :rtype: dict

    Example::

        >>> from yolov8.utils.cli import yolo_train_param_schema, parse_hyperparam
        >>> schema = yolo_train_param_schema()
        >>> parse_hyperparam("device=0,1,2,3", schema)
        ('device', [0, 1, 2, 3])
        >>> parse_hyperparam("name=run_2024_04", schema)
        ('name', 'run_2024_04')
    """
    from ultralytics.cfg import (
        CFG_BOOL_KEYS,
        CFG_FLOAT_KEYS,
        CFG_FRACTION_KEYS,
        CFG_INT_KEYS,
    )

    schema: dict = {}
    for k in CFG_INT_KEYS:
        schema[k] = int
    for k in CFG_FLOAT_KEYS:
        schema[k] = Union[int, float]
    for k in CFG_FRACTION_KEYS:
        schema[k] = fraction
    for k in CFG_BOOL_KEYS:
        schema[k] = bool

    # Special cases not covered by the plain CFG_*_KEYS sets.
    # Ordering inside Union matters: ``str`` accepts everything, so the
    # ``list[...]`` member must come first so that ``device=0,1,2,3``
    # parses to ``[0, 1, 2, 3]`` rather than the literal string
    # ``'0,1,2,3'``. The list-vs-scalar disambiguation in
    # :func:`_coerce_typed` then skips the list branch when the input
    # contains no comma, so ``device=cpu`` still falls through to the
    # ``str`` member.
    schema["device"] = Union[List[Union[int, str]], int, str]
    schema["imgsz"] = Union[List[int], int]
    schema["classes"] = List[int]
    schema["freeze"] = Union[List[int], int]
    schema["name"] = str
    schema["project"] = str
    schema["model"] = str
    schema["data"] = str
    schema["resume"] = Union[bool, str]
    schema["pretrained"] = Union[bool, str]
    schema["optimizer"] = str
    schema["tracker"] = str
    schema["cfg"] = str
    return schema


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
