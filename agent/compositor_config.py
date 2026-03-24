"""Backward-compatibility shim. Import from agent.brand.compositor_config instead."""
import importlib as _il, sys as _sys
_mod = _il.import_module("agent.brand.compositor_config")
_sys.modules[__name__] = _mod
