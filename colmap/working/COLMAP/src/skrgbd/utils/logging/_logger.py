import logging
import sys
import traceback

try:
    import IPython
except ModuleNotFoundError:
    pass


STDERR_LEVEL = (logging.WARNING + logging.ERROR) // 2


class Logger(logging.Logger):
    def __init__(self, name, level=logging.NOTSET):
        super().__init__(name, level)
        self.debug = RShiftLogFn(self.debug)
        self.info = RShiftLogFn(self.info)

    @classmethod
    def prepare_logger(cls, loglevel='debug', logger_id='sk_rgbd', logfile=None):
        loglevel = Logger.get_log_level(loglevel)
        logging.setLoggerClass(cls)
        logger = logging.getLogger(logger_id)
        logger.setLevel(loglevel)

        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(loglevel)
        stream_handler.setFormatter(logger.formatter)
        logger.addHandler(stream_handler)

        if logfile is not None:
            logger.add_logfile(logfile, loglevel)
        return logger

    @staticmethod
    def get_log_level(loglevel):
        if isinstance(loglevel, str):
            loglevel = {
                'critical': 50,
                'error': 40,
                'warning': 30,
                'info': 20,
                'debug': 10
            }[loglevel]

        if loglevel >= 50:
            return logging.CRITICAL
        elif loglevel >= 40:
            return logging.ERROR
        elif loglevel >= 30:
            return logging.WARNING
        elif loglevel >= 20:
            return logging.INFO
        elif loglevel >= 10:
            return logging.DEBUG
        elif loglevel >= 0:
            return [logging.WARNING, logging.INFO, logging.DEBUG][min(loglevel, 2)]
        else:
            return logging.NOTSET

    @property
    def formatter(self):
        return logging.Formatter(fmt='%(asctime)s %(levelname).1s %(filename)s:%(lineno)d  %(message)s')

    def add_logfile(self, file, loglevel='debug'):
        loglevel = Logger.get_log_level(loglevel)
        file_handler = logging.FileHandler(file)
        file_handler.setLevel(loglevel)
        file_handler.setFormatter(self.formatter)
        self.addHandler(file_handler)

    def log(self, loglevel, msg, *args, **kwargs):
        # LEGACY
        if (not isinstance(msg, str)) or (loglevel == STDERR_LEVEL):
            return super().log(loglevel, msg, *args, **kwargs)
        if 'extra' in kwargs:
            class_name = kwargs['extra']['context_name']
        else:  # legacy
            class_name, _, msg = msg.partition(': ')

        msg = f'{class_name:>14}:   ' + msg
        return super().log(loglevel, msg, *args, **kwargs)

    def prepare_for_scanning(self, logfile):
        self.removeHandler(self.handlers[0])
        logging.addLevelName(STDERR_LEVEL, 'STDERR')
        sys.stderr = StreamToLogger(self, STDERR_LEVEL)
        self.add_logfile(logfile)

        def showtraceback(self, *args, **kwargs):
            traceback_lines = traceback.format_exception(*sys.exc_info())
            message = ''.join(traceback_lines)
            sys.stderr.write(message)
            orig_show_traceback(self, *args, **kwargs)
        orig_show_traceback = IPython.core.interactiveshell.InteractiveShell.showtraceback
        IPython.core.interactiveshell.InteractiveShell.showtraceback = showtraceback

    def get_context_logger(self, context_name):
        return ContextLogger(self, dict(context_name=context_name))


class StreamToLogger(object):
    def __init__(self, logger, log_level):
        self.logger = logger
        self.log_level = log_level
        self.msg_line = ''

    def write(self, buf):
        for line in buf.splitlines(True):
            self.msg_line = self.msg_line + line
            if line.endswith(('\n', '\r')):
                self.logger.log(self.log_level, self.msg_line.rstrip())
                self.msg_line = ''

    def flush(self):
        pass


class RShiftLogFn:
    def __init__(self, log_fn):
        self._log_fn = log_fn

    def __call__(self, msg, *args, **kwargs):
        kwargs['stacklevel'] = kwargs.get('stacklevel', 1) + 1
        self._log_fn(msg, *args, **kwargs)

    def __rrshift__(self, msg):
        return self._log_fn(msg, stacklevel=2)


class ContextLogger(logging.LoggerAdapter):
    def __init__(self, logger, extra):
        self.logger = logger
        self.extra = extra
        self.debug = RShiftLogFn(self._debug)
        self.info = RShiftLogFn(self._info)

    def _debug(self, msg, *args, **kwargs):
        return self.log(logging.DEBUG, msg, *args, **kwargs)

    def _info(self, msg, *args, **kwargs):
        return self.log(logging.INFO, msg, *args, **kwargs)


for _ in ['urx', 'ursecmon']:  # disable verbose urx logging
    logging.getLogger(_).setLevel(logging.WARNING)
logger = Logger.prepare_logger(loglevel='debug', logger_id='sk_rgbd')
