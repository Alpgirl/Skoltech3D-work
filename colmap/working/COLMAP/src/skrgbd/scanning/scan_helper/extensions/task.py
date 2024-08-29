from time import sleep

from skrgbd.utils.logging import logger
from skrgbd.utils.parallel import get_timestamp, PropagatingThread as Thread


log_context = 'ScanHelper'


class Task:
    def __init__(self, target, desc=''):
        self.prior_tasks = []
        self.endtime = None
        self.desc = desc

        def _target():
            for task in self.prior_tasks:
                if not isinstance(task, Noop):
                    # logger.debug(f'WAIT: {self.desc} <<< {task.desc}')
                    task.wait()
            ret = target(self)
            self.endtime = get_timestamp()
            return ret

        self.thread = Thread(target=_target)

    def after(self, *tasks):
        self.prior_tasks = tasks
        return self

    def start(self):
        r"""Starts the task.

        Returns
        -------
        self
        """
        self.thread.start()
        return self

    def wait(self, delay_after=None):
        r"""Wait until the task terminates.

        Parameters
        ----------
        delay_after : float
            If not None, sleep `delay_after` seconds after the task terminates.

        Returns
        -------
        self
        """
        self.thread.join()
        if delay_after is not None:
            sleep(delay_after)
        return self

    def wait_after(self, delay, tag=None):
        delay = float(delay)
        suffix = f' #{tag}' if tag else ''
        desc = f'Wait after {self.desc}{suffix}'

        def target(task):
            self.thread.join()
            logger.debug(f'{log_context}: {desc}')
            sleep(max(.01, self.endtime + delay - get_timestamp()))
            logger.debug(f'{log_context}: {desc} DONE')
        return Task(target, desc)


class Noop:
    def wait(self):
        return self
