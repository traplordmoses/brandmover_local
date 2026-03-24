"""Backward-compatibility shim. Import from agent.learning.pref_extractor instead."""
import importlib as _il, sys as _sys
_mod = _il.import_module("agent.learning.pref_extractor")
_sys.modules[__name__] = _mod
