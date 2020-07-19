import argparse
import os

import numpy as np

from runeval import train_folders

from librubiks.train import TrainData
from librubiks.analysis import AnalysisData
from librubiks.solving.evaluation import EvalData

import librubiks.plots.trainplot as tp
import librubiks.plots.evalplot as ep

from librubiks.utils import Logger


def _get_args() -> (str, bool, bool, bool):
    parser = argparse.ArgumentParser()
    parser.add_argument("--location", nargs=1, type=str)
    parser.add_argument("--no-train", action="store_true")
    parser.add_argument("--no-analysis", action="store_true")
    parser.add_argument("--no-eval", action="store_true")
    args = parser.parse_args()
    args.location = args.location or train_folders[-1]
    return args.location, not args.no_train, not args.no_analysis, not args.no_eval


def _list_dirs(loc: str) -> list:
    return [x[0] for x in os.walk(loc)]


def _filter_by_name(name: str, directories: list) -> list:
    """
    Return parent folders of all directories with given name
    """
    dirs = list()
    for directory in directories:
        split = os.path.split(directory)
        if name in split:
            dirs.append(os.path.join(*split[:split.index(name)]))
    return np.unique(dirs).tolist()


def _get_data(loc: str, train: bool, analysis: bool, eval_: bool) -> (dict, dict, dict):
    """
    Return three dicts (or None), the keys of which are locations and the values data
    First for train, second for analysis, and third for evaluation
    """
    directories   = _list_dirs(loc)
    train_dirs    = _filter_by_name(TrainData.subfolder, directories)
    analysis_dirs = _filter_by_name(AnalysisData.subfolder, directories)
    eval_dirs     = _filter_by_name(EvalData.subfolder, directories)

    train_data    = [TrainData.load(x)    for x in train_dirs   ] if train    else None
    analysis_data = [AnalysisData.load(x) for x in analysis_dirs] if analysis else None
    eval_data     = [EvalData.load(x)     for x in eval_dirs    ] if eval_    else None

    return train_data, analysis_data, eval_data


def _make_plots(location: str, train: bool, analysis: bool, eval_: bool):
    train_data, analysis_data, eval_data = _get_data(location, train, analysis, eval_)


if __name__ == "__main__":
    args = _get_args()
    _make_plots(*args)
