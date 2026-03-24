"""Backward-compatibility shim. Import from agent.quality.brand_alignment instead."""
import importlib as _il, sys as _sys
_mod = _il.import_module("agent.quality.brand_alignment")
_sys.modules[__name__] = _mod
