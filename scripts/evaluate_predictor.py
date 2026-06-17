#!/usr/bin/env python
"""Thin wrapper: tad evaluate-predictor (Section 19)."""
from tad.cli import main

if __name__ == "__main__":
    import sys
    main(["evaluate-predictor", *sys.argv[1:]])
