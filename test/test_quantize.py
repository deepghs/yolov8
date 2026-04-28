"""Unit tests for ``yolov8.quantize`` — the pieces that don't need
to actually run quantization (which would pull onnxruntime + cv2
and take 5+ minutes per model).

Covers:

* Public surface (constants + Tier S dict) is intact.
* Head-exclude extraction works on a synthetic ONNX with mixed op
  types — verifies both the window size and the op-type filter.
* Names resolution prefers ``labels.json`` over the pt fallback.
* Error path: missing ``args.yaml`` raises
  :class:`QuantizationConfigError` with a useful message.
* CLI surface exposes the documented flags.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import onnx
import pytest
from onnx import helper, TensorProto

from yolov8.quantize import (
    DEFAULT_CALIB_N,
    DEFAULT_CALIB_SEED,
    HEAD_EXCLUDE_OPS,
    HEAD_TAIL_K,
    QuantizationConfigError,
    TIER_S,
    _read_names_for_threshold,
    cli as quantize_cli,
    collect_head_excludes,
    quantize_workdir,
)


# ---------------------------------------------------------------------------
# Public surface invariants
# ---------------------------------------------------------------------------

@pytest.mark.unittest
class TestPublicSurface:
    def test_head_tail_k_is_60(self):
        # Bumping this from 60 changes the architecture-wide guarantee
        # documented in plans/YOLO-INT8-PTQ-CALIBRATION-RECIPE.md §7.3
        # — re-run that experiment before shipping a new value.
        assert HEAD_TAIL_K == 60

    def test_head_exclude_covers_v8_v11_and_v10(self):
        v8_v11_core = {'Sigmoid', 'Softmax', 'Concat', 'Split', 'Reshape',
                       'Transpose', 'Sub', 'Add', 'Div', 'Mul'}
        v10_extras = {'TopK', 'GatherElements', 'GatherND', 'ReduceMax',
                      'Tile', 'Unsqueeze', 'Sign', 'Equal', 'Not', 'Mod',
                      'Cast', 'And'}
        assert v8_v11_core <= HEAD_EXCLUDE_OPS
        assert v10_extras <= HEAD_EXCLUDE_OPS

    def test_default_calib_n_is_128(self):
        assert DEFAULT_CALIB_N == 128
        assert DEFAULT_CALIB_SEED == 0

    def test_tier_s_recipe_keys(self):
        # These keys are written into the INT8 ONNX metadata_props as
        # dghs.yolov8.quant.* — renaming any of them is a wire-format
        # break.
        for k in ('calibrator', 'percentile', 'symmetric', 'per_channel',
                  'reduce_range', 'quant_format', 'recipe_name'):
            assert k in TIER_S, f'missing Tier S key: {k!r}'
        assert TIER_S['calibrator'] == 'Percentile'
        assert TIER_S['percentile'] == 99.999
        assert TIER_S['symmetric'] is True
        assert TIER_S['per_channel'] is True
        assert TIER_S['reduce_range'] is True


# ---------------------------------------------------------------------------
# Head-exclude extraction
# ---------------------------------------------------------------------------

def _make_synthetic_onnx(out_path: Path, op_types: list[str]) -> Path:
    """Build a minimal valid ONNX with the requested op-type sequence.

    Each node is a dummy single-input single-output node named
    ``node_<i>`` so we can verify both the tail window and the
    op-type filter.
    """
    inputs = [helper.make_tensor_value_info('x_0', TensorProto.FLOAT, [1])]
    outputs = []
    nodes = []
    for i, op in enumerate(op_types):
        nodes.append(helper.make_node(
            op,
            inputs=[f'x_{i}'],
            outputs=[f'x_{i+1}'],
            name=f'node_{i}',
        ))
    outputs = [helper.make_tensor_value_info(
        f'x_{len(op_types)}', TensorProto.FLOAT, [1])]
    graph = helper.make_graph(nodes, 'g', inputs, outputs)
    model = helper.make_model(graph,
                              opset_imports=[helper.make_opsetid('', 14)])
    onnx.save(model, str(out_path))
    return out_path


@pytest.mark.unittest
class TestCollectHeadExcludes:
    def test_only_tail_window_considered(self, tmp_path):
        # 100 nodes, all Sigmoid (an HEAD_EXCLUDE_OPS member). Only the
        # last HEAD_TAIL_K (=60) should be returned.
        p = _make_synthetic_onnx(tmp_path / 'm.onnx', ['Sigmoid'] * 100)
        excluded = collect_head_excludes(p)
        assert len(excluded) == HEAD_TAIL_K
        # they should be the LAST 60 nodes by name
        assert excluded == [f'node_{i}' for i in range(40, 100)]

    def test_only_excluded_op_types_picked(self, tmp_path):
        # Mixed tail: every other op is Conv (NOT excluded).
        ops = ['Conv', 'Sigmoid'] * 30  # 60 nodes
        p = _make_synthetic_onnx(tmp_path / 'm.onnx', ops)
        excluded = collect_head_excludes(p)
        assert len(excluded) == 30  # only the Sigmoids
        for n in excluded:
            assert n.startswith('node_')

    def test_v10_extras_picked_up(self, tmp_path):
        # Smoke-test that the v10 NMS-free postproc op types are in
        # the excluded set when present in the tail.
        ops = ['TopK', 'GatherElements', 'Mod', 'Conv', 'Conv']
        p = _make_synthetic_onnx(tmp_path / 'm.onnx', ops)
        excluded = collect_head_excludes(p)
        # tail-60 with 5 nodes -> all 5 in window; 3 excluded ops.
        assert set(excluded) == {'node_0', 'node_1', 'node_2'}

    def test_empty_tail_returns_empty(self, tmp_path):
        # All Conv (no excluded op types) -> empty exclude list.
        p = _make_synthetic_onnx(tmp_path / 'm.onnx', ['Conv'] * 70)
        assert collect_head_excludes(p) == []


# ---------------------------------------------------------------------------
# Names resolution (predator-init-free path)
# ---------------------------------------------------------------------------

@pytest.mark.unittest
class TestReadNamesForThreshold:
    def test_labels_json_list_is_indexed(self, tmp_path):
        labels = ['cat', 'dog', 'bird']
        (tmp_path / 'labels.json').write_text(json.dumps(labels),
                                              encoding='utf-8')
        names = _read_names_for_threshold(tmp_path / 'best.pt', tmp_path)
        assert names == {0: 'cat', 1: 'dog', 2: 'bird'}

    def test_labels_json_dict_keys_coerced_to_int(self, tmp_path):
        # Some pipelines emit dict-shaped labels.json (string keys).
        (tmp_path / 'labels.json').write_text(
            json.dumps({'0': 'a', '1': 'b'}), encoding='utf-8')
        names = _read_names_for_threshold(tmp_path / 'best.pt', tmp_path)
        assert names == {0: 'a', 1: 'b'}

    def test_missing_labels_json_returns_empty_when_no_pt(self, tmp_path):
        # No labels.json AND no usable .pt -> {} (we never invoke
        # YOLO.names; the .pt branch swallows its exception).
        names = _read_names_for_threshold(tmp_path / 'best.pt', tmp_path)
        assert names == {}


# ---------------------------------------------------------------------------
# Config error path
# ---------------------------------------------------------------------------

@pytest.mark.unittest
class TestConfigErrors:
    def test_missing_args_yaml_raises(self, tmp_path):
        # Create a workdir without args.yaml. weights/best.pt isn't
        # the limiting factor — args.yaml is checked first.
        (tmp_path / 'weights').mkdir()
        (tmp_path / 'weights' / 'best.pt').write_bytes(b'placeholder')
        with pytest.raises(QuantizationConfigError) as excinfo:
            quantize_workdir(tmp_path)
        assert 'args.yaml' in str(excinfo.value)

    def test_explicit_data_overrides_missing_args(self, tmp_path):
        # Even with args.yaml absent, if the caller passes --data
        # explicitly, we should error on the (also missing) data path
        # rather than the args.yaml path.
        (tmp_path / 'weights').mkdir()
        (tmp_path / 'weights' / 'best.pt').write_bytes(b'placeholder')
        with pytest.raises(QuantizationConfigError) as excinfo:
            quantize_workdir(tmp_path,
                             data=str(tmp_path / 'nonexistent.yaml'))
        # Either the args.yaml message OR the data-yaml-not-found
        # message is acceptable; both are config-error states.
        msg = str(excinfo.value)
        assert ('args.yaml' in msg) or ('not found' in msg)


# ---------------------------------------------------------------------------
# CLI surface
# ---------------------------------------------------------------------------

@pytest.mark.unittest
class TestCliSurface:
    def test_cli_has_documented_flags(self):
        names = {p.name for p in quantize_cli.params}
        for required in ('workdir', 'data', 'train_images', 'calib_n',
                         'calib_seed', 'imgsz', 'opset_version',
                         'no_eval', 'force'):
            assert required in names, f'missing CLI flag: --{required}'

    def test_calib_n_default_matches_module_constant(self):
        for p in quantize_cli.params:
            if p.name == 'calib_n':
                assert p.default == DEFAULT_CALIB_N
                return
        pytest.fail('calib_n not found in CLI params')
