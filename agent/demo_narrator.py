"""Backward-compatibility shim. Import from agent.video.demo_narrator instead."""
import importlib as _il, sys as _sys
_mod = _il.import_module("agent.video.demo_narrator")
_sys.modules[__name__] = _mod
