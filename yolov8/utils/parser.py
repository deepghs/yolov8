"""Schema-driven parser for ``KEY=VALUE`` style CLI flags.

The parser is reused by every click CLI in this package - keeping it
in its own module rather than embedded in :mod:`yolov8.utils.cli`
makes it easier to import in non-CLI contexts (tests, config-file
loaders, etc.) and shrinks the cli module to the parts that actually
depend on click.

Typed schema vocabulary
-----------------------

Each schema entry can be one of:

* ``int`` / ``float`` / ``bool`` / ``str`` / ``type(None)``
    Atomic Python types. ``str`` is special: it short-circuits any
    auto-detection so the raw token survives verbatim. Use this when a
    value that *looks* like an int / float / bool should stay a string
    (e.g. ``name=2024-04-27`` should not become an int via the leading
    ``2024``). ``type(None)`` accepts ``null`` / ``none`` / empty
    string and returns ``None``.

* :class:`typing.List[T]` *or* ``list[T]`` (Python 3.9+ PEP 585)
    Comma-separated input, e.g. ``device=0,1,2,3``. Each element is
    parsed by ``T``.

* :class:`typing.Tuple[T1, T2, ...]` *or* ``tuple[T1, T2, ...]``
    Fixed-length comma-separated input. ``Tuple[int, ...]`` (variadic
    via ``Ellipsis``) is also accepted and behaves like ``List[int]``
    but returns a ``tuple``.

* :class:`typing.Set[T]` *or* ``set[T]`` /
  :class:`typing.FrozenSet[T]` *or* ``frozenset[T]``
    Comma-separated input deduplicated into a (frozen) set.

* :class:`typing.Union[A, B, ...]` *or* ``A | B`` (Python 3.10+ PEP 604)
    Try each member in order. List / Tuple / Set members are skipped
    when the input contains no comma so ``Union[List[int], int]``
    correctly returns ``5`` (int) for ``5`` and ``[1, 2, 3]`` (list)
    for ``1,2,3``. ``Optional[T]`` (i.e. ``Union[T, None]``) is the
    natural way to express "value or null".

* :class:`typing.Literal["a", "b", 1]`
    Enum-style choice. The raw string is matched against each
    candidate (string equality for str, ``T(raw) == option`` for
    int / float / bool); the matched option is returned verbatim.

* any callable ``f(raw: str) -> Any``
    Custom coercion. The shipped :func:`fraction` helper checks
    ``[0.0, 1.0]`` floats, mirroring Ultralytics'
    ``CFG_FRACTION_KEYS`` validation.

Without a schema entry the parser falls back to auto-detect:
``json.loads`` first (covers ``int`` / ``float`` / ``bool`` / ``null``
/ list / dict literals), then a plain ``str``.

Public surface
--------------

* :func:`parse_value` — coerce one raw string per a single schema spec.
* :func:`parse_hyperparam` — parse one ``KEY=VALUE`` token.
* :func:`parse_hyperparams` — reduce a sequence of ``KEY=VALUE`` tokens.
* :func:`hyperparam_callback` / :func:`hyperparam_callback_factory` —
  glue for a click ``-p`` repeatable option.
* :func:`fraction` — built-in callable for ``[0.0, 1.0]`` validation.
* :func:`yolo_train_param_schema` — the schema for ``model.train(...)``
  built from ``ultralytics.cfg.CFG_*_KEYS``.
"""
from __future__ import annotations

import json
import re
import sys
import typing
from typing import Any, Callable, List, Mapping, Optional, Sequence, Tuple, Union

import click
from click.core import Context, Parameter

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


_VALID_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_BOOL_TRUE = {"true", "1", "yes", "on"}
_BOOL_FALSE = {"false", "0", "no", "off"}
_NULL_LITERALS = {"null", "none", ""}


class ParseError(ValueError):
    """Raised when a value cannot be coerced to its schema spec.

    Wrapped into :class:`click.BadParameter` by the higher-level CLI
    callbacks so users see a clean usage error rather than a raw
    traceback.
    """


# ---------------------------------------------------------------------------
# Built-in coercion callables
# ---------------------------------------------------------------------------

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

        >>> from yolov8.utils.parser import fraction
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
    if origin is tuple:
        args = typing.get_args(spec)
        if not args:
            return "tuple"
        if len(args) == 2 and args[1] is Ellipsis:
            return f"tuple[{_type_name(args[0])}, ...]"
        return f"tuple[{', '.join(_type_name(a) for a in args)}]"
    if origin in (set, frozenset):
        args = typing.get_args(spec)
        return f"{origin.__name__}[{_type_name(args[0]) if args else 'any'}]"
    if origin is typing.Literal:
        return "literal[" + ", ".join(repr(a) for a in typing.get_args(spec)) + "]"
    if origin in _UNION_ORIGINS:
        return " | ".join(_type_name(a) for a in typing.get_args(spec))
    if callable(spec):
        return getattr(spec, "__name__", repr(spec))
    return repr(spec)


def _auto_detect(raw: str) -> Any:
    """No-schema fallback: try ``json.loads`` first, fall back to the
    raw string. ``json`` covers int / float / bool / null / list / dict
    literals — exactly the cases Ultralytics' keyword interface
    handles natively."""
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


# ---------------------------------------------------------------------------
# parse_value: coerce one raw string per one schema spec
# ---------------------------------------------------------------------------

def parse_value(raw: str, spec: Any = None) -> Any:
    """Coerce a raw string per a single schema spec.

    Lower-level primitive used by :func:`parse_hyperparam`. Useful in
    its own right for callers loading typed values from non-CLI
    sources (env vars, INI configs, JSON-on-the-wire that needs
    re-validation, ...).

    Recursion: ``list[T]`` walks each comma-separated element through
    :func:`parse_value` against ``T``; ``Union[A, B]`` tries each
    member in order until one succeeds. Schema entries from
    :mod:`typing` (``typing.List[int]`` / ``typing.Union[...]``) and
    PEP 585 / PEP 604 native generics (``list[int]`` / ``int | str``,
    Python 3.9+ / 3.10+ respectively) are accepted equivalently
    because :func:`typing.get_origin` collapses them to the same
    runtime tags.

    :param raw: The raw string to coerce.
    :type raw: str
    :param spec: A schema spec — a Python type, a :mod:`typing` /
        PEP 585 / PEP 604 generic, or a callable. ``None`` (the
        default) routes through the JSON-or-string auto-detect path.
    :type spec: Any
    :returns: The coerced Python value.
    :rtype: Any
    :raises ParseError: When the coercion fails for any reason
        (wrong type, out-of-range numeric, no Union member matches,
        ...).
    :raises TypeError: When ``spec`` is not a recognised type / generic /
        callable.

    Example::

        >>> from yolov8.utils.parser import parse_value
        >>> parse_value("16", int)
        16
        >>> parse_value("0,1,2,3", list[int])  # PEP 585 syntax
        [0, 1, 2, 3]
        >>> parse_value("hello")               # no schema -> string
        'hello'
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

    # Generic typing constructs. ``get_origin`` normalises both the
    # ``typing.*`` forms and the PEP 585 / PEP 604 native ones.
    origin = typing.get_origin(spec)

    # list[T] / typing.List[T]
    if origin is list:
        item_args = typing.get_args(spec)
        item_spec = item_args[0] if item_args else None
        # ``-p k=`` (empty value) is a 0-element list, not [None].
        if raw.strip() == "":
            return []
        items = [s.strip() for s in raw.split(",")]
        return [parse_value(s, item_spec) for s in items]

    # tuple[X, Y, ...] / typing.Tuple[...]
    if origin is tuple:
        args = typing.get_args(spec)
        if raw.strip() == "":
            return ()
        items = [s.strip() for s in raw.split(",")]
        # Variadic: tuple[T, ...]
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(parse_value(s, args[0]) for s in items)
        # Bare ``tuple`` with no type args -> auto-detect each element.
        if not args:
            return tuple(_auto_detect(s) for s in items)
        # Fixed length.
        if len(items) != len(args):
            raise ParseError(
                f"tuple expected {len(args)} elements, got {len(items)} "
                f"in {raw!r}"
            )
        return tuple(parse_value(s, t) for s, t in zip(items, args))

    # set[T] / frozenset[T] / typing.Set[T] / typing.FrozenSet[T]
    if origin in (set, frozenset):
        item_args = typing.get_args(spec)
        item_spec = item_args[0] if item_args else None
        if raw.strip() == "":
            return origin()
        items = [s.strip() for s in raw.split(",")]
        decoded = [parse_value(s, item_spec) for s in items]
        return origin(decoded)

    # typing.Literal[...] enum-style choice.
    if origin is typing.Literal:
        return _coerce_literal(raw, typing.get_args(spec))

    # Union / PEP 604 ``A | B``
    if origin in _UNION_ORIGINS:
        last_err: Optional[Exception] = None
        members = typing.get_args(spec)
        # Disambiguation: when a Union member is a comma-collection
        # (list / tuple / set / frozenset) but the input has no comma,
        # skip it. This makes ``Union[int, str, list[int]]`` behave
        # intuitively: ``cpu`` -> 'cpu' (str), ``0`` -> 0 (int),
        # ``0,1,2,3`` -> [0, 1, 2, 3] (list). Without this, the bare
        # ``str`` branch always wins and ``0,1,2,3`` would become the
        # literal string ``'0,1,2,3'``.
        skip_collections = "," not in raw
        for member in members:
            if skip_collections and typing.get_origin(member) in _COLLECTION_ORIGINS:
                continue
            try:
                return parse_value(raw, member)
            except (ParseError, ValueError, TypeError) as err:
                last_err = err
        raise ParseError(
            f"value {raw!r} matches none of {_type_name(spec)}"
        ) from last_err

    # Custom callable parser (last resort - put after the typing-origin
    # checks so ``typing.Literal`` etc. don't get caught by the
    # ``callable(spec)`` branch first).
    if callable(spec):
        try:
            return spec(raw)
        except (ParseError, ValueError, TypeError) as err:
            raise ParseError(
                f"value {raw!r} rejected by {_type_name(spec)}: {err}"
            ) from err

    raise TypeError(f"unsupported schema type: {spec!r}")


#: Origin tags for "comma-separated collection" types. Used by the
#: Union disambiguation: when input has no comma, every member with an
#: origin in this set is skipped so the scalar member wins.
_COLLECTION_ORIGINS = (list, tuple, set, frozenset)


def _coerce_literal(raw: str, options) -> Any:
    """Match ``raw`` against a :class:`typing.Literal` option set.

    Each option is tried in order:

    * ``str`` options are matched by direct string equality.
    * ``bool`` / ``int`` / ``float`` options are matched after coercing
      ``raw`` through the option's type.
    * ``None`` options are matched against the null literals (``null``
      / ``none`` / empty).
    """
    for option in options:
        if option is None:
            if raw.strip().lower() in _NULL_LITERALS:
                return None
            continue
        if isinstance(option, bool):
            try:
                if _parse_bool(raw) == option:
                    return option
            except ParseError:
                pass
            continue
        if isinstance(option, (int, float)):
            try:
                if type(option)(raw) == option:
                    return option
            except (ValueError, TypeError):
                pass
            continue
        if isinstance(option, str):
            if raw == option:
                return option
            continue
    raise ParseError(
        f"value {raw!r} not in literal options "
        f"{tuple(options)!r}"
    )


# ---------------------------------------------------------------------------
# parse_hyperparam: parse one KEY=VALUE token
# ---------------------------------------------------------------------------

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
      ``device=0,1,2,3``. Each element goes through :func:`parse_value`
      with ``T``.
    * Schema entry ``Union[A, B, ...]``: try in order.

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

        >>> from yolov8.utils.parser import parse_hyperparam
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
        decoded = parse_value(value, spec)
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

        >>> from yolov8.utils.parser import parse_hyperparams
        >>> schema = {"batch": int, "device": list[int]}
        >>> parse_hyperparams(["batch=16", "device=0,1"], schema)
        {'batch': 16, 'device': [0, 1]}
    """
    out: dict = {}
    for tok in tokens:
        k, v = parse_hyperparam(tok, schema=schema)
        out[k] = v
    return out


# ---------------------------------------------------------------------------
# Click glue
# ---------------------------------------------------------------------------

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

        >>> from yolov8.utils.parser import hyperparam_callback_factory
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


# ---------------------------------------------------------------------------
# YOLO train schema, derived from Ultralytics' own typing constants
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

    Plus a handful of special cases the plain CFG sets don't cover:

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

        >>> from yolov8.utils.parser import yolo_train_param_schema, parse_hyperparam
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
    # :func:`parse_value` then skips the list branch when the input
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
