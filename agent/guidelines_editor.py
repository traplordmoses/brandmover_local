"""Backward-compatibility shim. Import from agent.brand.guidelines_editor instead."""
import importlib as _il, sys as _sys
_mod = _il.import_module("agent.brand.guidelines_editor")
_sys.modules[__name__] = _mod
