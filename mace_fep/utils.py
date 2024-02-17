import time
import contextlib
import functools
import logging
from typing import Optional, Union
import sys
import os
import argparse
import logging


logger = logging.getLogger("mace_fep")


def parse_arguments():

    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str)
    parser.add_argument("--replicas", type=int, default=4)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--steps_per_iter", type=int)
    parser.add_argument("--iters", type=int)

    parser.add_argument(
        "--model_path", type=str, default="input_files/SPICE_sm_inv_neut_E0_swa.model"
    )
    parser.add_argument("-o", "--output", type=str, default="junk")
    parser.add_argument("--minimise", action="store_true")
    parser.add_argument("--restart", action="store_true")
    parser.add_argument("--log_level", type=str, default="INFO")
    parser.add_argument("--ligA_idx", type=int, help="open interval [0, ligA_idx) selects the ligand atoms for ligA", default=None)
    parser.add_argument("--ligB_idx", type=int, help="open interval [ligA_idx, ligB_idx) selects the ligand atoms for ligB", default=None)
    parser.add_argument("--ligA_const", help="atom to constrain in ligA", type=int)
    parser.add_argument("--ligB_const", help="atom to constrain in ligB", default=None, type=int)
    parser.add_argument("--mode", choices=["NEQ", "EQ"], default="NEQ")
    parser.add_argument("--dtype", type=str, default="float64", choices=["float32", "float64"])
    parser.add_argument("--no_mixing", action="store_true")
    parser.add_argument("--reverse", action="store_true")
    parser.add_argument("--equilibrate", action="store_true")
    parser.add_argument("--report_interval", type=int, default=100)
    parser.add_argument("--use_ssc", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    return parser



def setup_logger(
    level: Union[int, str] = logging.INFO,
    tag: Optional[str] = None,
    directory: Optional[str] = None,
):
    logger = logging.getLogger()
    logger.setLevel(level)

    formatter = logging.Formatter(
        "%(asctime)s.%(msecs)03d %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if (directory is not None) and (tag is not None):
        os.makedirs(name=directory, exist_ok=True)
        path = os.path.join(directory, tag + ".log")
        fh = logging.FileHandler(path)
        fh.setFormatter(formatter)

        logger.addHandler(fh)



@contextlib.contextmanager
def time_it(task_name):
    """Context manager to log execution time of a block of code.

    Parameters
    ----------
    task_name : str
        The name of the task that will be reported.

    """
    timer = Timer()
    timer.start(task_name)
    yield timer  # Resume program
    timer.stop(task_name)
    timer.report_timing()


def with_timer(task_name):
    """Decorator that logs the execution time of a function.

    Parameters
    ----------
    task_name : str
        The name of the task that will be reported.

    """

    def _with_timer(func):
        @functools.wraps(func)
        def _wrapper(*args, **kwargs):
            with time_it(task_name):
                return func(*args, **kwargs)

        return _wrapper

    return _with_timer


class Timer(object):
    """A class with stopwatch-style timing functions.

    Examples
    --------
    >>> timer = Timer()
    >>> timer.start('my benchmark')
    >>> for i in range(10):
    ...     pass
    >>> elapsed_time = timer.stop('my benchmark')
    >>> timer.start('second benchmark')
    >>> for i in range(10):
    ...     for j in range(10):
    ...         pass
    >>> elsapsed_time = timer.stop('second benchmark')
    >>> timer.report_timing()

    """

    def __init__(self):
        self.reset_timing_statistics()

    def reset_timing_statistics(self, benchmark_id=None):
        """Reset the timing statistics.

        Parameters
        ----------
        benchmark_id : str, optional
            If specified, only the timings associated to this benchmark
            id will be reset, otherwise all timing information are.

        """
        if benchmark_id is None:
            self._t0 = {}
            self._t1 = {}
            self._completed = {}
        else:
            self._t0.pop(benchmark_id, None)
            self._t1.pop(benchmark_id, None)
            self._completed.pop(benchmark_id, None)

    def start(self, benchmark_id):
        """Start a timer with given benchmark_id."""
        self._t0[benchmark_id] = time.time()

    def stop(self, benchmark_id):
        try:
            t0 = self._t0[benchmark_id]
        except KeyError:
            logger.warning("Can't stop timing for {}".format(benchmark_id))
        else:
            self._t1[benchmark_id] = time.time()
            elapsed_time = self._t1[benchmark_id] - t0
            self._completed[benchmark_id] = elapsed_time
            return elapsed_time

    def partial(self, benchmark_id):
        """Return the elapsed time of the given benchmark so far."""
        try:
            t0 = self._t0[benchmark_id]
        except KeyError:
            logger.warning("Couldn't return partial timing for {}".format(benchmark_id))
        else:
            return time.time() - t0

    def report_timing(self, clear=True):
        """Log all the timings at the debug level.

        Parameters
        ----------
        clear : bool
            If True, the stored timings are deleted after being reported.

        Returns
        -------
        elapsed_times : dict
            The dictionary benchmark_id : elapsed time for all benchmarks.

        """
        for benchmark_id, elapsed_time in self._completed.items():
            logger.debug("{} took {:8.3f}s".format(benchmark_id, elapsed_time))

        if clear is True:
            self.reset_timing_statistics()
