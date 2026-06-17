#!/usr/bin/env python
"""Thin wrapper: tad train-predictor (Section 19)."""
from tad.cli import main

if __name__ == "__main__":
    import sys
    main(["train-predictor", *sys.argv[1:]])
