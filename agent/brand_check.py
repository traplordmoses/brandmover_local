"""Backward-compatibility shim. Import from agent.quality.brand_check instead."""
import importlib as _il, sys as _sys
_mod = _il.import_module("agent.quality.brand_check")
_sys.modules[__name__] = _mod
