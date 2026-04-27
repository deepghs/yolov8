"""Unit tests for ``yolov8.utils.cli`` hyperparameter parsing.

Covers:

* auto-detect (no schema): JSON literals + string fallback
* per-type coercion (int, float, bool, str, fraction)
* schema-forced ``str`` defeats accidental int coercion (``at=111``
  with ``{'at': str}`` returns ``'111'``)
* comma-separated list parsing with sub-typing
* ``Union`` members tried in order, with list-vs-scalar
  disambiguation by comma-presence
* error reporting via :class:`click.BadParameter`
* the YOLO train schema built from ult's ``CFG_*_KEYS``
"""
from __future__ import annotations

from typing import List, Union

import click
import pytest

# Import directly from the cli module rather than ``yolov8.utils`` so
# this test file is loadable on minimal Python installs (no torch /
# numpy / pandas) - the parser is the only thing we exercise here, so
# it should run wherever click + pytest are available.
from yolov8.utils.cli import (
    ParseError,
    fraction,
    hyperparam_callback_factory,
    parse_hyperparam,
    parse_hyperparams,
    parse_yversion,
    yolo_train_param_schema,
)


# ---------------------------------------------------------------------------
# parse_hyperparam: auto-detect (no schema)
# ---------------------------------------------------------------------------

@pytest.mark.unittest
class TestAutoDetect:
    @pytest.mark.parametrize("token, expected", [
        ("batch=16", ("batch", 16)),
        ("mosaic=1.0", ("mosaic", 1.0)),
        ("cos_lr=true", ("cos_lr", True)),
        ("cos_lr=false", ("cos_lr", False)),
        ("device=null", ("device", None)),
        ("optimizer=AdamW", ("optimizer", "AdamW")),
        ("name=my-run", ("name", "my-run")),
        ('label="hello world"', ("label", "hello world")),
        ("freeze=[1,2,3]", ("freeze", [1, 2, 3])),
    ])
    def test_known_shapes(self, token, expected):
        assert parse_hyperparam(token) == expected

    def test_string_fallback_for_unparseable_value(self):
        # A non-numeric, non-quoted value falls through to string.
        assert parse_hyperparam("foo=bar/baz") == ("foo", "bar/baz")


# ---------------------------------------------------------------------------
# parse_hyperparam: malformed input -> click.BadParameter
# ---------------------------------------------------------------------------

@pytest.mark.unittest
class TestMalformedInput:
    @pytest.mark.parametrize("token", [
        "no_equals_sign",
        "=missing_key",
        "1starts_with_digit=oops",
        "bad-key=value",  # hyphen not allowed in identifier
        "spaces in key=oops",
    ])
    def test_malformed(self, token):
        with pytest.raises(click.BadParameter):
            parse_hyperparam(token)


# ---------------------------------------------------------------------------
# parse_hyperparam: typed schema
# ---------------------------------------------------------------------------

@pytest.mark.unittest
class TestTypedSchema:
    def test_int_schema(self):
        assert parse_hyperparam("patience=20", {"patience": int}) \
            == ("patience", 20)

    def test_int_schema_rejects_non_int(self):
        with pytest.raises(click.BadParameter, match="int"):
            parse_hyperparam("patience=abc", {"patience": int})

    def test_float_schema(self):
        assert parse_hyperparam("lr0=0.01", {"lr0": float}) \
            == ("lr0", 0.01)

    def test_str_schema_forces_string(self):
        # Bare auto-detect would coerce '111' -> 111. Schema=str
        # short-circuits that.
        assert parse_hyperparam("at=111", {"at": str}) == ("at", "111")
        assert parse_hyperparam("name=2024-04-27", {"name": str}) \
            == ("name", "2024-04-27")

    def test_bool_schema_truthy(self):
        for raw in ["true", "True", "1", "yes", "ON"]:
            assert parse_hyperparam(f"k={raw}", {"k": bool}) == ("k", True)

    def test_bool_schema_falsy(self):
        for raw in ["false", "False", "0", "no", "off"]:
            assert parse_hyperparam(f"k={raw}", {"k": bool}) == ("k", False)

    def test_bool_schema_rejects_garbage(self):
        with pytest.raises(click.BadParameter, match="bool"):
            parse_hyperparam("k=maybe", {"k": bool})

    def test_callable_schema(self):
        assert parse_hyperparam("mosaic=0.5", {"mosaic": fraction}) \
            == ("mosaic", 0.5)

    def test_callable_schema_validates(self):
        with pytest.raises(click.BadParameter, match="fraction"):
            parse_hyperparam("mosaic=2.0", {"mosaic": fraction})

    def test_unknown_key_falls_back_to_auto(self):
        # Schema is supplied but the key isn't in it -> auto-detect.
        out = parse_hyperparam("custom=42", {"patience": int})
        assert out == ("custom", 42)


# ---------------------------------------------------------------------------
# parse_hyperparam: list[T] schema
# ---------------------------------------------------------------------------

@pytest.mark.unittest
class TestListSchema:
    def test_list_int(self):
        assert parse_hyperparam("classes=0,1,2,3", {"classes": List[int]}) \
            == ("classes", [0, 1, 2, 3])

    def test_list_int_single_element(self):
        # Plain List[T] (not inside a Union) accepts a single element.
        assert parse_hyperparam("classes=5", {"classes": List[int]}) \
            == ("classes", [5])

    def test_list_int_empty(self):
        # ``-p classes=`` -> [].
        assert parse_hyperparam("classes=", {"classes": List[int]}) \
            == ("classes", [])

    def test_list_int_rejects_non_int_element(self):
        with pytest.raises(click.BadParameter, match="int"):
            parse_hyperparam("classes=0,foo,2", {"classes": List[int]})

    def test_list_str(self):
        assert parse_hyperparam("tags=a,b,c", {"tags": List[str]}) \
            == ("tags", ["a", "b", "c"])

    def test_nested_list_union(self):
        # list[Union[int, str]] sub-parses each element.
        spec = {"d": List[Union[int, str]]}
        assert parse_hyperparam("d=0,cpu,1", spec) \
            == ("d", [0, "cpu", 1])


# ---------------------------------------------------------------------------
# parse_hyperparam: Union schema with list disambiguation
# ---------------------------------------------------------------------------

@pytest.mark.unittest
class TestUnionSchema:
    def test_union_picks_first_match(self):
        # Try int first, fall back to str.
        spec = {"x": Union[int, str]}
        assert parse_hyperparam("x=42", spec) == ("x", 42)
        assert parse_hyperparam("x=abc", spec) == ("x", "abc")

    def test_union_with_list_skips_list_when_no_comma(self):
        # Union[List[int], int] + ``5`` -> 5 (skip list, no comma).
        spec = {"x": Union[List[int], int]}
        assert parse_hyperparam("x=5", spec) == ("x", 5)

    def test_union_with_list_picks_list_on_comma(self):
        spec = {"x": Union[List[int], int]}
        assert parse_hyperparam("x=1,2,3", spec) == ("x", [1, 2, 3])

    def test_device_style_full_disambiguation(self):
        spec = {"device": Union[List[Union[int, str]], int, str]}
        assert parse_hyperparam("device=0", spec) == ("device", 0)
        assert parse_hyperparam("device=cpu", spec) == ("device", "cpu")
        assert parse_hyperparam("device=0,1,2,3", spec) \
            == ("device", [0, 1, 2, 3])
        assert parse_hyperparam("device=cpu,cuda", spec) \
            == ("device", ["cpu", "cuda"])


# ---------------------------------------------------------------------------
# parse_hyperparams (sequence)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# YOLO train schema
# ---------------------------------------------------------------------------

@pytest.mark.unittest
class TestYoloTrainSchema:
    @pytest.fixture(scope="class")
    def schema(self):
        # The schema is built from constants in ultralytics.cfg, so
        # this whole class is skipped on environments that don't have
        # ultralytics installed (e.g. minimal CI envs that just want
        # to verify parse_hyperparam typing).
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
        # name/project/model/data are forced str so '2024' isn't auto-int.
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

    def test_unknown_key_in_train_schema_falls_back_to_auto(self, schema):
        # Even with the train schema active, unknown keys auto-detect.
        assert parse_hyperparam("ultra_custom=42", schema) \
            == ("ultra_custom", 42)


# ---------------------------------------------------------------------------
# hyperparam_callback_factory
# ---------------------------------------------------------------------------

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
# parse_yversion (sanity)
# ---------------------------------------------------------------------------

@pytest.mark.unittest
class TestParseYVersion:
    def test_digit_string_to_int(self):
        assert parse_yversion("11") == 11
        assert parse_yversion("8") == 8

    def test_named_passes_through(self):
        assert parse_yversion("rtdetr") == "rtdetr"
        assert parse_yversion("world") == "world"

    def test_int_passes_through(self):
        assert parse_yversion(11) == 11

    def test_whitespace_stripped(self):
        assert parse_yversion("  12  ") == 12
