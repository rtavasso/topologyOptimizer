#!/usr/bin/env python
"""Thin wrapper: tad build-dataset (Section 19)."""
from tad.cli import main

if __name__ == "__main__":
    import sys
    main(["build-dataset", *sys.argv[1:]])
