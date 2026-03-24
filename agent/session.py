"""Backward-compatibility shim. Import from agent.learning.session instead."""
import importlib as _il, sys as _sys
_mod = _il.import_module("agent.learning.session")
_sys.modules[__name__] = _mod
