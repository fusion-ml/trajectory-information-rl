"""
Timing utilities.
"""

import time
import datetime
import logging


class Timer(object):
    """
    Timer class. Thanks to Eli Bendersky, Josiah Yoder, Jonas Adler, Can Kavaklıoğlu,
    and others from https://stackoverflow.com/a/50957722.
    """

    def __init__(self, name=None, filename=None, level=None):
        self.name = name
        self.filename = filename
        self.level = logging.INFO if level is None else level

    def __enter__(self):
        self.tstart = time.time()

    def __exit__(self, type, value, traceback):
        message = "Elapsed: %.2f seconds" % (time.time() - self.tstart)
        if self.name:
            message = "*[TIME] [%s] " % self.name + message
        logging.log(self.level, message)
        if self.filename:
            with open(self.filename, "a") as file:
                print(str(datetime.datetime.now()) + ": ", message, file=file)
