"""Backward-compatibility shim. Import from agent.scheduling.topic_bank instead."""
import importlib as _il, sys as _sys
_mod = _il.import_module("agent.scheduling.topic_bank")
_sys.modules[__name__] = _mod
