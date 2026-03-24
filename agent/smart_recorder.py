"""Backward-compatibility shim. Import from agent.video.smart_recorder instead."""
import importlib as _il, sys as _sys
_mod = _il.import_module("agent.video.smart_recorder")
_sys.modules[__name__] = _mod
