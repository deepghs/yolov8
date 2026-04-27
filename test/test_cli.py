"""Unit tests for ``yolov8.utils.cli`` (CLI-specific helpers).

The bulk of the parsing / typed-schema machinery lives in
:mod:`yolov8.utils.parser`; its tests are in ``test_parser.py``. This
file only covers the click-glue bits that stay in
:mod:`yolov8.utils.cli`:

* :func:`parse_yversion` (string / int normalisation)
* re-exports from :mod:`yolov8.utils.parser` keep working under the
  old import path (``from yolov8.utils.cli import parse_hyperparam``)
"""
from __future__ import annotations

import pytest

# These imports must succeed - the cli module re-exports the parser
# surface for back-compat with code that imported from cli before the
# parser was extracted.
from yolov8.utils.cli import (  # noqa: F401
    ParseError,
    fraction,
    hyperparam_callback,
    hyperparam_callback_factory,
    parse_hyperparam,
    parse_hyperparams,
    parse_value,
    parse_yversion,
    print_version,
    yolo_train_param_schema,
)


# ---------------------------------------------------------------------------
# parse_yversion (CLI-only helper)
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


# ---------------------------------------------------------------------------
# Re-export sanity (back-compat with the pre-extraction import path)
# ---------------------------------------------------------------------------

@pytest.mark.unittest
class TestReexports:
    """Code written against the old ``from yolov8.utils.cli import
    parse_hyperparam`` path must keep working after parser was
    extracted into its own module."""

    def test_parse_hyperparam_via_cli_module(self):
        from yolov8.utils.cli import parse_hyperparam as cli_pp
        from yolov8.utils.parser import parse_hyperparam as par_pp
        assert cli_pp is par_pp
        assert cli_pp("batch=16") == ("batch", 16)

    def test_yolo_train_param_schema_via_cli_module(self):
        from yolov8.utils.cli import yolo_train_param_schema as cli_s
        from yolov8.utils.parser import yolo_train_param_schema as par_s
        assert cli_s is par_s
