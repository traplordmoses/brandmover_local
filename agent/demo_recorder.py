"""Backward-compatibility shim. Import from agent.video.demo_recorder instead."""
import importlib as _il, sys as _sys
_mod = _il.import_module("agent.video.demo_recorder")
_sys.modules[__name__] = _mod
