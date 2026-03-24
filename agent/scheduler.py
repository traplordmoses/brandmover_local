"""Backward-compatibility shim. Import from agent.scheduling.scheduler instead."""
import importlib as _il, sys as _sys
_mod = _il.import_module("agent.scheduling.scheduler")
_sys.modules[__name__] = _mod
