"""Backward-compatibility shim. Import from agent.learning.feedback instead."""
import importlib as _il, sys as _sys
_mod = _il.import_module("agent.learning.feedback")
_sys.modules[__name__] = _mod
