import os
import sys


CODE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_ROOT = os.path.join(CODE_ROOT, "src")
for path in (CODE_ROOT, SRC_ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)
