import os

# AITER repo root proxy: make 'aiter' importable when repo root is on sys.path.
# The real Python package lives under aiter/aiter/.
__path__.append(os.path.join(os.path.dirname(__file__), "aiter"))
from .aiter import *
