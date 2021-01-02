import argparse
from typing import Tuple

from runeval import train_folders

from librubiks.jobs import PlotJob
from pelutils import log


def _get_args() -> Tuple[str, bool, bool, bool]:
    parser = argparse.ArgumentParser()
    parser.add_argument("location")
    parser.add_argument("--no-train", action="store_true")
    parser.add_argument("--no-analysis", action="store_true")
    parser.add_argument("--no-eval", action="store_true")
    args = parser.parse_args()
    args.location = args.location or train_folders[-1]
    return args.location, not args.no_train, not args.no_analysis, not args.no_eval


if __name__ == "__main__":
    with log.log_errors:
        args = _get_args()
        job = PlotJob(*args)
        job.execute()
