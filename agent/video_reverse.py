"""Backward-compatibility shim. Import from agent.video.video_reverse instead."""
import importlib as _il, sys as _sys
_mod = _il.import_module("agent.video.video_reverse")
_sys.modules[__name__] = _mod
