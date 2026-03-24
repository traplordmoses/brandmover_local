"""Backward-compatibility shim. Import from agent.scheduling.heartbeat instead."""
import importlib as _il, sys as _sys
_mod = _il.import_module("agent.scheduling.heartbeat")
_sys.modules[__name__] = _mod
