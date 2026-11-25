"""Backward compatibility alias: larch_cli_wrapper -> md_exafs.

The package itself must stay a real package (not ``sys.modules`` aliased to
``md_exafs``): otherwise ``larch_cli_wrapper.pipeline`` is resolved through
``md_exafs.__path__`` and loads a second copy of ``md_exafs/pipeline.py``,
which also rebinds ``md_exafs.pipeline`` to that copy.  The per-module shims in
this directory alias the real submodules.
"""

from md_exafs import *  # noqa: F403
from md_exafs import DEFAULT_CACHE_DIR, __all__, __version__  # noqa: F401
