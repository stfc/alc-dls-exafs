"""Backward compatibility alias: debye_waller_core -> debye_waller."""

import sys

import md_exafs.debye_waller as _mod

sys.modules[__name__] = _mod
