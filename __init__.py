"""Compatibility proxy for Stage0 namespace collision avoidance."""
import os
import sys
import importlib

_proxy_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Remove entries that cause the proxy to be found before PYTHONPATH.
for _entry in list(sys.path):
    if _entry == _proxy_dir:
        sys.path.remove(_entry)
    if _entry == "" and os.getcwd() == _proxy_dir:
        sys.path.remove("")
        sys.path.append("")

# Re-import aiter from the real location (PYTHONPATH must include the repo root).
if "aiter" in sys.modules:
    del sys.modules["aiter"]
_real_aiter = importlib.import_module("aiter")
sys.modules[__name__] = _real_aiter
