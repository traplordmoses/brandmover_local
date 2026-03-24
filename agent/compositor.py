"""Backward-compatibility shim. Import from agent.brand.compositor instead."""
import importlib as _il, sys as _sys
_mod = _il.import_module("agent.brand.compositor")
_sys.modules[__name__] = _mod
