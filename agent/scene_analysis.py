"""Backward-compatibility shim. Import from agent.video.scene_analysis instead."""
import importlib as _il, sys as _sys
_mod = _il.import_module("agent.video.scene_analysis")
_sys.modules[__name__] = _mod
