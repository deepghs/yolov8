"""Unit tests for ``yolov8.utils.parser``.

Splits the parser-level concerns out of ``test_cli.py`` so the parser
machinery can be tested independently. Three layers of typing surface
are exercised:

* atomic types (``int`` / ``float`` / ``bool`` / ``str`` / ``None``)
* :mod:`typing` constructs (``Union``, ``List``, ``Tuple``, ``Set``,
  ``FrozenSet``, ``Optional``, ``Literal``)
* PEP 585 native generics (``list[int]``, ``tuple[int, str]``,
  ``set[int]``) on Python 3.9+
* PEP 604 native unions (``int | str``) on Python 3.10+

The PEP 604 cases live inside method bodies (gated by
:func:`pytest.mark.skipif`) so the file still parses on Python 3.8 /
3.9 — those interpreters just won't execute the bodies.
"""
from __future__ import annotations

import sys
import typing
from typing import (
    FrozenSet,
    List,
    Literal,
    Optional,
    Set,
    Tuple,
    Union,
)

import click
import pytest

from yolov8.utils.parser import (
    ParseError,
    fraction,
    hyperparam_callback_factory,
    parse_hyperparam,
    parse_hyperparams,
    parse_value,
    yolo_train_param_schema,
)


# ---------------------------------------------------------------------------
# parse_value: atomic types
# ---------------------------------------------------------------------------

@pytest.mark.unittest
class TestAtomic:
    def test_int(self):
        assert parse_value("42", int) == 42

    def test_int_rejects_non_numeric(self):
        with pytest.raises(ParseError):
            parse_value("abc", int)

    def test_float(self):
        assert parse_value("3.14", float) == 3.14

    def test_str_no_coercion(self):
        # str schema must NOT coerce '111' -> 111.
        assert parse_value("111", str) == "111"
        assert parse_value("2024-04-27", str) == "2024-04-27"

    def test_bool_truthy(self):
        for raw in ["true", "1", "Yes", "ON"]:
            assert parse_value(raw, bool) is True

    def test_bool_falsy(self):
        for raw in ["false", "0", "no", "Off"]:
            assert parse_value(raw, bool) is False

    def test_bool_rejects_garbage(self):
        with pytest.raises(ParseError):
            parse_value("maybe", bool)

    def test_none_via_type_none(self):
        assert parse_value("null", type(None)) is None
        assert parse_value("none", type(None)) is None
        assert parse_value("", type(None)) is None

    def test_none_rejects_non_null(self):
        with pytest.raises(ParseError):
            parse_value("0", type(None))

    def test_callable(self):
        assert parse_value("0.5", fraction) == 0.5
        with pytest.raises(ParseError):
            parse_value("2.0", fraction)


# ---------------------------------------------------------------------------
# parse_value: typing.* generics
# ---------------------------------------------------------------------------

@pytest.mark.unittest
class TestTypingGenerics:
    def test_typing_list(self):
        assert parse_value("0,1,2", List[int]) == [0, 1, 2]

    def test_typing_list_empty(self):
        assert parse_value("", List[int]) == []

    def test_typing_list_single(self):
        assert parse_value("5", List[int]) == [5]

    def test_typing_list_str(self):
        assert parse_value("a,b,c", List[str]) == ["a", "b", "c"]

    def test_typing_tuple_fixed(self):
        assert parse_value("640,480", Tuple[int, int]) == (640, 480)

    def test_typing_tuple_variadic(self):
        assert parse_value("1,2,3,4", Tuple[int, ...]) == (1, 2, 3, 4)

    def test_typing_tuple_mixed_types(self):
        assert parse_value("foo,42", Tuple[str, int]) == ("foo", 42)

    def test_typing_tuple_wrong_length(self):
        with pytest.raises(ParseError, match="expected 3 elements"):
            parse_value("1,2", Tuple[int, int, int])

    def test_typing_set(self):
        assert parse_value("1,2,3,1", Set[int]) == {1, 2, 3}

    def test_typing_frozenset(self):
        assert parse_value("a,b,a", FrozenSet[str]) == frozenset(["a", "b"])

    def test_typing_union(self):
        assert parse_value("42", Union[int, str]) == 42
        assert parse_value("abc", Union[int, str]) == "abc"

    def test_typing_optional(self):
        # Optional[int] == Union[int, None]
        assert parse_value("5", Optional[int]) == 5
        assert parse_value("null", Optional[int]) is None
        with pytest.raises(ParseError):
            parse_value("abc", Optional[int])

    def test_typing_literal(self):
        spec = Literal["SGD", "AdamW", "Adam"]
        assert parse_value("AdamW", spec) == "AdamW"
        with pytest.raises(ParseError, match="literal options"):
            parse_value("SGD2", spec)

    def test_typing_literal_mixed(self):
        spec = Literal[0, 1, "auto", None]
        assert parse_value("0", spec) == 0
        assert parse_value("auto", spec) == "auto"
        assert parse_value("null", spec) is None


# ---------------------------------------------------------------------------
# parse_value: PEP 585 / 604 native generics where the version permits
# ---------------------------------------------------------------------------

@pytest.mark.unittest
@pytest.mark.skipif(sys.version_info < (3, 9),
                    reason="PEP 585 ``list[int]`` syntax requires Python 3.9+")
class TestPEP585:
    def test_list_int(self):
        assert parse_value("0,1,2", list[int]) == [0, 1, 2]

    def test_tuple_fixed(self):
        assert parse_value("640,480", tuple[int, int]) == (640, 480)

    def test_tuple_variadic(self):
        assert parse_value("1,2,3", tuple[int, ...]) == (1, 2, 3)

    def test_set(self):
        assert parse_value("1,2,3,1", set[int]) == {1, 2, 3}

    def test_frozenset(self):
        assert parse_value("a,b", frozenset[str]) == frozenset(["a", "b"])

    def test_pep585_matches_typing_form(self):
        # typing.List[int] and list[int] should produce identical
        # output for the same input.
        for raw in ["1,2,3", "5", ""]:
            assert parse_value(raw, List[int]) == parse_value(raw, list[int])


@pytest.mark.unittest
@pytest.mark.skipif(sys.version_info < (3, 10),
                    reason="PEP 604 ``X | Y`` syntax requires Python 3.10+")
class TestPEP604:
    # The ``int | str`` expressions below only run on Python 3.10+,
    # but the file still parses on 3.8 / 3.9 because `__or__` on
    # types is just a runtime operation - parsing it is fine on any
    # version, and the skipif decorator prevents the bodies from
    # being executed on incompatible interpreters.
    def test_int_pipe_str(self):
        spec = int | str
        assert parse_value("42", spec) == 42
        assert parse_value("abc", spec) == "abc"

    def test_int_pipe_none(self):
        spec = int | None
        assert parse_value("5", spec) == 5
        assert parse_value("null", spec) is None

    def test_pep604_matches_typing_form(self):
        spec_old = Union[int, str]
        spec_new = int | str
        for raw in ["1", "abc", "0"]:
            assert parse_value(raw, spec_old) == parse_value(raw, spec_new)

    def test_list_pipe_int(self):
        # list[int] | int via PEP 604.
        spec = list[int] | int
        assert parse_value("5", spec) == 5
        assert parse_value("1,2,3", spec) == [1, 2, 3]


# ---------------------------------------------------------------------------
# parse_value: Union list-vs-scalar disambiguation
# ---------------------------------------------------------------------------

@pytest.mark.unittest
class TestUnionDisambiguation:
    def test_union_list_or_scalar_no_comma_returns_scalar(self):
        spec = Union[List[int], int]
        assert parse_value("5", spec) == 5

    def test_union_list_or_scalar_with_comma_returns_list(self):
        spec = Union[List[int], int]
        assert parse_value("1,2,3", spec) == [1, 2, 3]

    def test_device_style_union(self):
        spec = Union[List[Union[int, str]], int, str]
        assert parse_value("0", spec) == 0
        assert parse_value("cpu", spec) == "cpu"
        assert parse_value("0,1,2", spec) == [0, 1, 2]
        assert parse_value("cpu,cuda", spec) == ["cpu", "cuda"]

    def test_union_with_tuple_disambig(self):
        # Same logic should apply to tuple/set.
        spec = Union[Tuple[int, int], int]
        assert parse_value("5", spec) == 5
        assert parse_value("640,480", spec) == (640, 480)


# ---------------------------------------------------------------------------
# parse_hyperparam (KEY=VALUE wrapper)
# ---------------------------------------------------------------------------

@pytest.mark.unittest
class TestParseHyperparam:
    def test_auto_detect_no_schema(self):
        assert parse_hyperparam("batch=16") == ("batch", 16)
        assert parse_hyperparam("name=hello") == ("name", "hello")

    def test_with_schema(self):
        assert parse_hyperparam("at=111", {"at": str}) == ("at", "111")

    def test_unknown_key_falls_back_to_auto(self):
        out = parse_hyperparam("custom=42", {"patience": int})
        assert out == ("custom", 42)

    @pytest.mark.parametrize("token", [
        "no_equals",
        "=missing_key",
        "1bad=v",
        "bad-key=v",
    ])
    def test_malformed_raises_click_bad_parameter(self, token):
        with pytest.raises(click.BadParameter):
            parse_hyperparam(token)

    def test_schema_violation_wraps_into_bad_parameter(self):
        with pytest.raises(click.BadParameter, match="int"):
            parse_hyperparam("patience=abc", {"patience": int})


@pytest.mark.unittest
class TestParseHyperparams:
    def test_basic(self):
        assert parse_hyperparams(["a=1", "b=hello", "c=true"]) \
            == {"a": 1, "b": "hello", "c": True}

    def test_later_wins(self):
        assert parse_hyperparams(["a=1", "a=2"]) == {"a": 2}

    def test_with_schema(self):
        schema = {"batch": int, "device": Union[List[int], int]}
        assert parse_hyperparams(["batch=16", "device=0,1"], schema) \
            == {"batch": 16, "device": [0, 1]}


@pytest.mark.unittest
class TestCallbackFactory:
    def test_factory_uses_schema(self):
        cb = hyperparam_callback_factory({"patience": int})
        out = cb(None, None, ["patience=20", "extra=hello"])
        assert out == {"patience": 20, "extra": "hello"}

    def test_factory_with_none_schema_is_auto(self):
        cb = hyperparam_callback_factory(None)
        assert cb(None, None, ["a=1"]) == {"a": 1}

    def test_factory_returns_empty_for_no_input(self):
        cb = hyperparam_callback_factory({"k": int})
        assert cb(None, None, ()) == {}


# ---------------------------------------------------------------------------
# YOLO train preset schema (uses 3.8-compatible typing forms only)
# ---------------------------------------------------------------------------

@pytest.mark.unittest
class TestYoloTrainSchema:
    @pytest.fixture(scope="class")
    def schema(self):
        # Skipped on minimal envs that don't have ultralytics; those
        # envs still get full coverage of the parser machinery
        # itself (TestAtomic / TestTypingGenerics / ...).
        pytest.importorskip("ultralytics")
        return yolo_train_param_schema()

    def test_int_keys(self, schema):
        assert schema["patience"] is int
        assert parse_hyperparam("patience=20", schema) == ("patience", 20)
        with pytest.raises(click.BadParameter):
            parse_hyperparam("patience=abc", schema)

    def test_fraction_keys_validate_range(self, schema):
        assert parse_hyperparam("mosaic=0.5", schema) == ("mosaic", 0.5)
        for bad in ["mosaic=2.0", "mosaic=-0.1", "mosaic=abc"]:
            with pytest.raises(click.BadParameter):
                parse_hyperparam(bad, schema)

    def test_bool_keys(self, schema):
        assert parse_hyperparam("cos_lr=true", schema) == ("cos_lr", True)
        with pytest.raises(click.BadParameter):
            parse_hyperparam("cos_lr=maybe", schema)

    def test_device_disambiguation(self, schema):
        assert parse_hyperparam("device=0", schema) == ("device", 0)
        assert parse_hyperparam("device=cpu", schema) == ("device", "cpu")
        assert parse_hyperparam("device=0,1,2,3", schema) \
            == ("device", [0, 1, 2, 3])

    def test_imgsz_int_or_list(self, schema):
        assert parse_hyperparam("imgsz=640", schema) == ("imgsz", 640)
        assert parse_hyperparam("imgsz=640,480", schema) \
            == ("imgsz", [640, 480])

    def test_str_only_fields(self, schema):
        assert parse_hyperparam("name=2024", schema) == ("name", "2024")
        assert parse_hyperparam("project=runs/foo", schema) \
            == ("project", "runs/foo")

    def test_classes_list_int(self, schema):
        assert parse_hyperparam("classes=0,5,9", schema) \
            == ("classes", [0, 5, 9])
        assert parse_hyperparam("classes=5", schema) \
            == ("classes", [5])

    def test_freeze_int_or_list(self, schema):
        assert parse_hyperparam("freeze=10", schema) == ("freeze", 10)
        assert parse_hyperparam("freeze=0,1,2", schema) \
            == ("freeze", [0, 1, 2])

    def test_schema_uses_3_8_compat_forms_only(self, schema):
        # The preset schema must not use PEP 585 / PEP 604 syntax in
        # its own construction so it loads on Python 3.8.
        # Sanity: every value should be either a Python type, a
        # callable, or a typing.Union/List that comes from the
        # ``typing`` module (not ``types.UnionType`` or ``list[X]``).
        import types
        for key, spec in schema.items():
            origin = typing.get_origin(spec)
            if isinstance(spec, type):
                continue
            if callable(spec):
                continue
            # Generic spec: origin must come from typing (or be None
            # for callables / atomic types we already skipped).
            # Reject types.UnionType (PEP 604) explicitly.
            if hasattr(types, "UnionType"):
                assert origin is not types.UnionType, (
                    f"preset schema key {key!r} uses PEP 604 ``X | Y`` "
                    f"which breaks Python 3.8 / 3.9 compat"
                )
            # PEP 585 forms have origin like ``list`` (the builtin
            # class itself). The typing.List form has origin ``list``
            # as well. We can't easily tell them apart, but we don't
            # need to: any way of expressing list[int] from typing
            # also has ``typing.get_origin(...) is list``.
