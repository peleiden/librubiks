import argparse
import os

import numpy as np

from runeval import train_folders

from librubiks.jobs import PlotJob


def _get_args() -> (str, bool, bool, bool):
    parser = argparse.ArgumentParser()
    parser.add_argument("--location", type=str)
    parser.add_argument("--no-train", action="store_true")
    parser.add_argument("--no-analysis", action="store_true")
    parser.add_argument("--no-eval", action="store_true")
    args = parser.parse_args()
    args.location = args.location or train_folders[-1]
    return args.location, not args.no_train, not args.no_analysis, not args.no_eval


if __name__ == "__main__":
    args = _get_args()
    job = PlotJob(*args)
    job.execute()
