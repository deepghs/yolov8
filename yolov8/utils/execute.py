import threading


def delayed_execution(func, delay_seconds: int = 2):
    timer = threading.Timer(delay_seconds, func)
    timer.daemon = True
    timer.start()
    return timer
