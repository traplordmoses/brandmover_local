"""Backward-compatibility shim. Import from agent.brand.font_manager instead."""
import importlib as _il, sys as _sys
_mod = _il.import_module("agent.brand.font_manager")
_sys.modules[__name__] = _mod
