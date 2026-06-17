#!/usr/bin/env python
"""Thin wrapper: tad make-report (Section 19)."""
from tad.cli import main

if __name__ == "__main__":
    import sys
    main(["make-report", *sys.argv[1:]])
