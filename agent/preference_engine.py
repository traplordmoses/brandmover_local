"""Backward-compatibility shim. Import from agent.learning.preference_engine instead."""
import importlib as _il, sys as _sys
_mod = _il.import_module("agent.learning.preference_engine")
_sys.modules[__name__] = _mod
