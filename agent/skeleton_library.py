"""Backward-compatibility shim. Import from agent.learning.skeleton_library instead."""
import importlib as _il, sys as _sys
_mod = _il.import_module("agent.learning.skeleton_library")
_sys.modules[__name__] = _mod
