from datetime import datetime
from threading import Event, Thread


class PropagatingThread(Thread):
    r"""Source https://stackoverflow.com/a/31614591"""
    def run(self):
        self.exc = None
        try:
            self.ret = self._target(*self._args, **self._kwargs)
        except BaseException as e:
            self.exc = e

    def join(self, timeout=None):
        super().join(timeout)
        if self.exc:
            raise self.exc
        return self.ret


class ThreadSet:
    def __init__(self, threads):
        self.threads = threads

    def start(self):
        for thread in self.threads:
            thread.start()

    def join(self, timeout=None):
        for thread in self.threads:
            thread.join(timeout)
            if timeout is not None:
                if thread.is_alive():
                    return

    def start_and_join(self, timeout=None):
        self.start()
        self.join(timeout)

    def is_alive(self):
        return any(thread.is_alive() for thread in self.threads)


class EventVar(Event):
    """TODO"""

    def __init__(self):
        super().__init__()
        del self._flag
        self._var = None

    def is_set(self):
        return self._var is not None

    def set(self, val):
        with self._cond:
            self._var = val
            self._cond.notify_all()

    def clear(self):
        with self._cond:
            self._var = None

    def wait(self, timeout=None):
        with self._cond:
            if not self.is_set():
                self._cond.wait(timeout)
            return self._var


def get_timestamp():
    return datetime.now().timestamp()
