"""Utility helpers re-exported at the top of :mod:`yolov8.utils`.

The CLI helpers from :mod:`yolov8.utils.cli` are loaded eagerly so
they're available from the very first import (they only depend on
``click`` and the standard library, which keeps the module loadable
on minimal envs that haven't installed torch / numpy / pandas yet).

Everything else - checkpoint introspection, markdown parsing, the
threshold extractor - is loaded lazily via :pep:`562`'s
``__getattr__``. The first attribute access on, say,
``yolov8.utils.compute_threshold_data`` imports
:mod:`yolov8.utils.threshold` and pulls in its numpy / ditk
dependencies; until then they stay un-imported. This means
``from yolov8.utils import parse_hyperparam`` works in a
click-only Python install.
"""
from importlib import import_module

# Eager: pure stdlib + click, cheap.
from .cli import (
    GLOBAL_CONTEXT_SETTINGS,
    parse_yversion,
    print_version,
)
from .parser import (
    ParseError,
    fraction,
    hyperparam_callback,
    hyperparam_callback_factory,
    parse_hyperparam,
    parse_hyperparams,
    parse_value,
    yolo_train_param_schema,
)

#: Mapping from a public attribute name to ``"<submodule>:<attr>"``;
#: read by :func:`__getattr__` to materialise the symbol on first
#: access. Keep this list in sync with the public surface.
_LAZY_ATTRS = {
    'derive_model_meta':                          'ckpt:derive_model_meta',
    'derive_model_meta_from_path':                'ckpt:derive_model_meta_from_path',
    'markdown_to_df':                             'md:markdown_to_df',
    'float_pe':                                   'pe:float_pe',
    'compute_threshold_data':                     'threshold:compute_threshold_data',
    'compute_threshold_data_from_trainer':        'threshold:compute_threshold_data_from_trainer',
    'compute_threshold_data_from_validator_stats': 'threshold:compute_threshold_data_from_validator_stats',
}


def __getattr__(name):
    """:pep:`562` lazy attribute resolver for the heavy submodules.

    :param name: Attribute name being looked up.
    :type name: str
    :returns: The resolved object from the relevant submodule.
    :raises AttributeError: When ``name`` is not in the lazy map.
    """
    if name in _LAZY_ATTRS:
        mod_name, attr = _LAZY_ATTRS[name].split(':')
        mod = import_module(f'.{mod_name}', package=__name__)
        return getattr(mod, attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    """Help IDE autocompletion: include lazy attrs in ``dir()``."""
    return sorted(set(globals()) | set(_LAZY_ATTRS))
