import datetime
import logging

from tqdm.auto import tqdm as _tqdm

from ._logger import logger


class TqdmHandler(logging.Handler):
    def __init__(self, level=logging.NOTSET):
        super().__init__(level)

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)
            self.flush()
        except (KeyboardInterrupt, SystemExit):
            raise
        except:
            self.handleError(record)


class tqdm(_tqdm):
    def __init__(self, *args, logger=logger, eta_format=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.eta_format = eta_format
        if not isnotebook():
            self._logger = logger
            self._default_handler = self._logger.handlers[0]
            tqdm_handler = TqdmHandler(level=self._default_handler.level)
            tqdm_handler.setFormatter(self._default_handler.formatter)
            self._logger.handlers[0] = tqdm_handler

    def __del__(self):
        if not isnotebook():
            self._logger.handlers[0] = self._default_handler
        super().__del__()

    def __iter__(self):
        if self.eta_format is None:
            return super().__iter__()
        else:
            if self.eta_format is True:
                def eta_format(now_t, remaining_t):
                    eta = now_t + remaining_t
                    eta = eta.strftime('%H:%M')
                    return f'ETA: {eta}'
                self.eta_format = eta_format
            for i in super().__iter__():
                if (self.format_dict['total'] is not None) and (self.format_dict['n'] is not None) and (self.format_dict['rate'] is not None):
                    now_t = datetime.datetime.now()
                    remaining_t = (self.format_dict['total'] - self.format_dict['n']) / self.format_dict['rate']
                    remaining_t = datetime.timedelta(seconds=remaining_t)
                    eta = self.eta_format(now_t, remaining_t)
                    self.bar_format = '{l_bar}{bar}{r_bar} ' + eta
                yield i


def isnotebook():
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter


class DummyTqdm:
    def __init__(self, iterator=None, *args, **kwargs):
        self.iterator = iterator

    def __iter__(self):
        return iter(self.iterator)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return

    def close(self):
        return

    def set_description(self, desc):
        return

    def update(self, *args, **kwargs):
        return
