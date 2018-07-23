import time


def estimates_function_runtime(function):
    """
    decorator for runtime estimates
    """

    def wrapped(*args, **kwargs):
        start = time.time()
        function_results = function(*args, **kwargs)
        end = time.time()
        return end - start, function_results

    return wrapped
