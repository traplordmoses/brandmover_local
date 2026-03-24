"""Backward-compatibility shim. Import from agent.scheduling.schedule_queue instead."""
import importlib as _il, sys as _sys
_mod = _il.import_module("agent.scheduling.schedule_queue")
_sys.modules[__name__] = _mod
