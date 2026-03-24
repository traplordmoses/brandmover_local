"""Backward-compatibility shim. Import from agent.publishing.channels instead."""
import importlib as _il, sys as _sys
_mod = _il.import_module("agent.publishing.channels")
_sys.modules[__name__] = _mod
