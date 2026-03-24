"""Backward-compatibility shim. Import from agent.video.video_styler instead."""
import importlib as _il, sys as _sys
_mod = _il.import_module("agent.video.video_styler")
_sys.modules[__name__] = _mod
