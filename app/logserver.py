"""Logserver implementation."""
from __future__ import absolute_import, division, unicode_literals

import logging
from logging.handlers import RotatingFileHandler

__all__ = 'configure_logger', 'parse_loglevel'
_DEFAULT_LOGFORMAT = (
    '%(asctime)s [%(levelname)s]: %(processName)s %(message)s')
# Since logging module doesn't standardize name-to-level conversion, here's a
# dict with the standard levels.
_NAME2LVL = {
    'CRITICAL': 50,
    'ERROR': 40,
    'WARNING': 30,
    'INFO': 20,
    'DEBUG': 10,
}

_logger = logging.getLogger(__name__)
_main_logger = logging.getLogger()


def configure_logger(logfile_name='service.log',
                     logfile_maxsize=1024**2,
                     logfile_maxbackups=10,
                     log_format=_DEFAULT_LOGFORMAT,
                     local_log_level='INFO'):
    """Configure the logger used by :class`LogServer` instances.

    This function must be called before any logging servers are started (not
    that there should ever be need for more than one) and cannot be called
    again. This is to ensure thread safety.

    :param logfile_name: Name of the file to which logs are written,
        defaults to 'service.log'.
    :type logfile_name: str, optional

    :param logfile_maxsize: Maximal size in bytes of a single rotating
        log file, defaults to 1 MiB.
    :type logfile_maxsize: int, optional

    :param logfile_maxbackups: Maximal number of backup log files kept,
        defaults to 10.
    :type logfile_maxbackups: int, optional

    :param log_format: Format string for log records output by this
        server. A reasonable default is provided.
    :type log_format: str, optional

    :param local_log_level: Name of the log level for loggers in *this*
        library. Refer to standard documentation for possible names. This
        setting does not affect loggers from workers, defaults to 'INFO'.
    :type local_log_level: str, optional
    :raises RuntimeError: when logger already configured.
    """
    _handler = RotatingFileHandler(logfile_name,
                                   maxBytes=logfile_maxsize,
                                   backupCount=logfile_maxbackups,
                                   encoding='utf-8',
                                   delay=True)
    fmter = logging.Formatter(log_format)
    _handler.setFormatter(fmter)
    _main_logger.addHandler(_handler)
    _main_logger.setLevel(parse_loglevel(local_log_level))

    stdhandler = logging.StreamHandler()
    stdhandler.setFormatter(fmter)
    _main_logger.addHandler(stdhandler)
    logging.root.setLevel("INFO")


def parse_loglevel(log_level):
    """Get logging level constant number from a string.

    If the string is an integer literal, return it as integer. Otherwise try to
    interpret the string as one of the standard level names and return value
    associated with that.

    :param log_level: String naming the log level, to be parsed.
    :type log_level: str

    :return: Integer value of the log level, as used by ``logging`` module.
    :rtype: int

    :raise KeyError: When ``log_level`` is neither an integer literal nor
        the name of a standard logging level.
    """
    try:
        lvlnum = int(log_level)
    except ValueError:
        lvlnum = _NAME2LVL[log_level.upper()]

    return lvlnum
