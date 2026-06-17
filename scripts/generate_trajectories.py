#!/usr/bin/env python
"""Thin wrapper: tad generate-trajectories (Section 19)."""
from tad.cli import main

if __name__ == "__main__":
    import sys
    main(["generate-trajectories", *sys.argv[1:]])
