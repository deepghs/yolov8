"""Unit tests for the path-anonymisation helpers in
:mod:`yolov8.export`.

These exercise the defensive shape handling: missing dicts, missing
keys, attribute-bearing objects (e.g. ult's ``IterableSimpleNamespace``)
versus plain dicts, non-string values, and bare model names that
*aren't* paths and therefore shouldn't be hashed.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from yolov8.export import (
    _anonymise_path_field,
    _anonymise_state_dict_paths,
)


# ---------------------------------------------------------------------------
# Single-field anonymiser
# ---------------------------------------------------------------------------

@pytest.mark.unittest
class TestAnonymisePathField:
    def test_dict_path_string_gets_hashed(self):
        d = {'data': '/nfs/x/y.yaml'}
        _anonymise_path_field(d, 'data')
        assert d['data'] != '/nfs/x/y.yaml'
        assert len(d['data']) == 56  # sha3-224 hex
        int(d['data'], 16)  # raises if not hex

    def test_object_attr_path_string_gets_hashed(self):
        o = SimpleNamespace(data='/data/runs/best.pt', other=42)
        _anonymise_path_field(o, 'data')
        assert o.data != '/data/runs/best.pt'
        assert len(o.data) == 56
        assert o.other == 42  # unrelated attrs untouched

    def test_dict_missing_field_is_skipped(self):
        d = {'project': '/some/path'}
        _anonymise_path_field(d, 'data')  # 'data' not present
        assert d == {'project': '/some/path'}

    def test_object_missing_attr_is_skipped(self):
        o = SimpleNamespace(project='/some/path')
        _anonymise_path_field(o, 'data')
        assert o.project == '/some/path'
        assert not hasattr(o, 'data')

    def test_none_container_is_skipped(self):
        # Should be a no-op rather than a crash
        _anonymise_path_field(None, 'data')

    def test_none_value_is_skipped(self):
        d = {'data': None}
        _anonymise_path_field(d, 'data')
        assert d == {'data': None}

    def test_empty_string_is_skipped(self):
        d = {'data': ''}
        _anonymise_path_field(d, 'data')
        assert d == {'data': ''}

    def test_non_path_string_is_skipped(self):
        # Bare 'yolo11n.pt' has no path separator; keep verbatim.
        d = {'model': 'yolo11n.pt'}
        _anonymise_path_field(d, 'model')
        assert d == {'model': 'yolo11n.pt'}

    def test_non_string_value_is_skipped(self):
        d = {'data': 42, 'project': ['/a', '/b']}
        _anonymise_path_field(d, 'data')
        _anonymise_path_field(d, 'project')
        assert d == {'data': 42, 'project': ['/a', '/b']}

    def test_windows_separator_triggers_hash(self):
        d = {'data': r'C:\Users\me\dataset\data.yaml'}
        _anonymise_path_field(d, 'data')
        assert d['data'] != r'C:\Users\me\dataset\data.yaml'
        assert len(d['data']) == 56

    def test_setattr_failure_is_swallowed(self):
        # An object that refuses setattr (e.g. __slots__ + readonly).
        # We accept that the value cannot be scrubbed there — the
        # dict-side scan is the primary defence.
        class Frozen:
            __slots__ = ()  # nothing settable

        # Just exercise the path; the implementation should not crash.
        _anonymise_path_field(Frozen(), 'data')


# ---------------------------------------------------------------------------
# Whole-state_dict walker
# ---------------------------------------------------------------------------

@pytest.mark.unittest
class TestAnonymiseStateDictPaths:
    def _hash_len(self, s):
        return isinstance(s, str) and len(s) == 56

    def test_walks_train_args_dict(self):
        sd = {
            'train_args': {
                'data': '/nfs/x/data.yaml',
                'project': '/data/runs',
                'model': 'yolo11n.pt',  # not a path
                'unrelated': 42,
            },
        }
        _anonymise_state_dict_paths(sd)
        assert self._hash_len(sd['train_args']['data'])
        assert self._hash_len(sd['train_args']['project'])
        # Bare 'yolo11n.pt' not anonymised.
        assert sd['train_args']['model'] == 'yolo11n.pt'
        assert sd['train_args']['unrelated'] == 42

    def test_walks_inner_model_args(self):
        # The leaker we just fixed: even when train_args is clean,
        # the model object's own .args dict carries the same paths.
        inner = SimpleNamespace(args={
            'data': '/nfs/y/data.yaml',
            'project': '/data/runs',
            'save_dir': '/data/runs/foo',
        })
        sd = {'model': inner}
        _anonymise_state_dict_paths(sd)
        assert self._hash_len(inner.args['data'])
        assert self._hash_len(inner.args['project'])
        assert self._hash_len(inner.args['save_dir'])

    def test_walks_inner_model_args_attribute_shape(self):
        # Older ult versions exposed args as IterableSimpleNamespace
        # rather than dict — same anon should apply via setattr.
        inner_args = SimpleNamespace(
            data='/nfs/z/data.yaml', project='/data/runs', save_dir='/data/x',
        )
        inner = SimpleNamespace(args=inner_args)
        sd = {'model': inner}
        _anonymise_state_dict_paths(sd)
        assert self._hash_len(inner.args.data)
        assert self._hash_len(inner.args.project)
        assert self._hash_len(inner.args.save_dir)

    def test_walks_ema_args(self):
        ema = SimpleNamespace(args={
            'data': '/nfs/ema/data.yaml',
            'project': '/data/runs',
        })
        sd = {'ema': ema}
        _anonymise_state_dict_paths(sd)
        assert self._hash_len(ema.args['data'])
        assert self._hash_len(ema.args['project'])

    def test_walks_ema_module_args(self):
        # Some ult versions wrap the EMA model under ema.module.
        inner = SimpleNamespace(args={
            'data': '/nfs/em/data.yaml',
        })
        ema = SimpleNamespace(module=inner)
        sd = {'ema': ema}
        _anonymise_state_dict_paths(sd)
        assert self._hash_len(inner.args['data'])

    def test_missing_train_args_is_skipped(self):
        sd = {}  # no train_args, no model, no ema
        _anonymise_state_dict_paths(sd)
        assert sd == {}

    def test_missing_inner_args_is_skipped(self):
        # model object exists but has no .args attribute
        sd = {'model': SimpleNamespace()}
        _anonymise_state_dict_paths(sd)  # should not crash

    def test_train_args_not_a_dict_is_skipped(self):
        # If train_args got serialised as a non-dict somehow, walker
        # leaves it alone.
        sentinel = object()
        sd = {'train_args': sentinel}
        _anonymise_state_dict_paths(sd)
        assert sd['train_args'] is sentinel

    def test_does_not_mutate_unrelated_fields(self):
        sd = {
            'train_args': {'data': '/x.yaml', 'patience': 20, 'imgsz': 640},
            'epoch': 50,
            'best_fitness': 0.95,
        }
        _anonymise_state_dict_paths(sd)
        assert self._hash_len(sd['train_args']['data'])
        assert sd['train_args']['patience'] == 20
        assert sd['train_args']['imgsz'] == 640
        assert sd['epoch'] == 50
        assert sd['best_fitness'] == 0.95

    def test_idempotent(self):
        # Running twice over an already-anonymised state_dict should
        # be a no-op (sha3-224 hex strings have no '/' or '\\').
        sd = {'train_args': {'data': '/nfs/x/data.yaml',
                              'project': '/data/runs'}}
        _anonymise_state_dict_paths(sd)
        snapshot = dict(sd['train_args'])
        _anonymise_state_dict_paths(sd)
        assert sd['train_args'] == snapshot
