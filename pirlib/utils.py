import json
import time as _time
from datetime import datetime


def find_by_id(iterable, id):
    for item in iterable:
        if item.id == id:
            return item


class PerformanceTimer(object):
    """
    This class is used to measure the execution time of a function. It contains 2 type metrics: Process time and
    Wall time.
    """

    def __init__(self, context_statement, output_dir):
        """
        :param Text context_statement: the statement to log
        """
        self._context_statement = context_statement
        self._start_wall_time = None
        self._start_process_time = None
        self._output_dir = output_dir

    def __enter__(self):
        print(
            "{} Entering timed context: {}".format(
                datetime.now().strftime("%Y-%m-%d %H:%M:%S.%F"), self._context_statement
            )
        )
        self._start_wall_time = _time.perf_counter()
        self._start_process_time = _time.process_time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        end_wall_time = _time.perf_counter()
        end_process_time = _time.process_time()
        elapsed_wall_time = end_wall_time - self._start_wall_time
        elapsed_process_time = end_process_time - self._start_process_time
        print(
            "{} Exiting timed context: {} [Wall Time: {}s, Process Time: {}s]".format(
                datetime.now().strftime("%Y-%m-%d %H:%M:%S.%F"),
                self._context_statement,
                elapsed_wall_time,
                elapsed_process_time,
            )
        )
        with (self._output_dir / "elapsed_time.json").open("w") as f:
            json.dump({"wall_time": elapsed_wall_time, "process_time": elapsed_process_time}, f)
