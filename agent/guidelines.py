"""Backward-compatibility shim. Import from agent.brand.guidelines instead."""
import importlib as _il, sys as _sys
_mod = _il.import_module("agent.brand.guidelines")
_sys.modules[__name__] = _mod
