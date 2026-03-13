"""Local NLU domain datasets (ontologies, eval sets).

Proxies the HuggingFace ``datasets`` package so that libraries like
accelerate can still do ``from datasets import IterableDataset``.
"""

import importlib as _importlib
import os as _os
import sys as _sys

# Temporarily remove this local package from sys.modules and hide the
# project root from sys.path so importlib finds the HuggingFace package.
_self_module = _sys.modules.pop(__name__)
_project_root = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_original_path = _sys.path[:]
_sys.path = [p for p in _sys.path if _os.path.abspath(p) != _project_root]

try:
    _hf = _importlib.import_module("datasets")
    for _attr in dir(_hf):
        if not _attr.startswith("_"):
            globals()[_attr] = getattr(_hf, _attr)
except ImportError:
    pass
finally:
    _sys.path = _original_path
    _sys.modules[__name__] = _self_module
