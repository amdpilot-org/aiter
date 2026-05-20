"""Stage0 import proxy for nested AITER source package."""
import importlib as _importlib
_real = _importlib.import_module(__name__ + '.aiter')
globals().update({k: v for k, v in _real.__dict__.items() if not (k.startswith('__') and k not in {'__version__', '__all__'})})
