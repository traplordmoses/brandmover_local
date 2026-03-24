"""Backward-compatibility shim. Import from agent.brand.figma instead."""
import importlib as _il, sys as _sys
_mod = _il.import_module("agent.brand.figma")
_sys.modules[__name__] = _mod
