from functools import wraps
import errno
import os
import signal
#
class TimeoutError(Exception):
    pass
#
def timeout(timeout, *args):
    def decorate(f):
        def handler(signum, frame):
            raise TimeoutError()

        def new_f(*args):
            signal.signal(signal.SIGALRM, handler)
            signal.alarm(timeout)
            return f(*args)
            signal.alarm(0)

        new_f.__name__ = f.__name__
        return new_f
    return decorate