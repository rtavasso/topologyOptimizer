#!/usr/bin/env python
"""Thin wrapper: tad run-online-optimizer (Section 19)."""
from tad.cli import main

if __name__ == "__main__":
    import sys
    main(["run-online-optimizer", *sys.argv[1:]])
