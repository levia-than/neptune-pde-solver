import sys

try:
    from . import _neptune_mlir as _backend
except ImportError as e:
    print(f"CRITICAL ERROR: Failed to import C++ backend: {e}") 
    import os
    print(f"Current working dir: {os.getcwd()}")
    _backend = None

class GlobalContext:
    def __init__(self):
        if _backend:
            self.compiler = _backend.Compiler()
        else:
            self.compiler = None
    
    def dump(self):
        if self.compiler:
            return self.compiler.dump()
        return "; Mock Dump"

_default_ctx = GlobalContext()

def get_compiler():
    return _default_ctx.compiler
