from functools import wraps
from time import perf_counter
from typing import Callable


def timeit(func: Callable) -> Callable:
    """
    Decorator for timing function execution time.

    Args:
        func: The function to time.
    Returns:
        The wrapped function.
    """

    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = perf_counter()
        result = func(*args, **kwargs)
        end_time = perf_counter()
        total_time = end_time - start_time
        print(f"Function {func.__name__} took {total_time:.2f} seconds")
        return result

    return timeit_wrapper


def runtime_alert(time_limit: int) -> Callable[[], Callable]:
    """
    Decorator with argument.
    """

    def inner_func(func: Callable) -> Callable:
        """
        Decorator that triggers and alert if the runtime of the function exceeds a time limit.

        Args:
            func: The function to time.
        Returns:
            The wrapped function.
        """

        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = perf_counter()
            result = func(*args, **kwargs)
            total_time = perf_counter() - start_time
            if total_time > time_limit:
                print(
                    f"Function {func.__name__} took more than {time_limit:.2f} seconds ({total_time:.2f} seconds)"
                )
            return result

        return wrapper

    return inner_func


def checkpoint(time_ref: float = perf_counter()) -> Callable[..., None]:
    """
    Closure that stores a time checkpoint that is updated at every call.
    Each call prints the time elapsed since the last checkpoint with a custom message.

    Args:
        time_ref: The time reference to start from. By default, the time of the call will be taken.
    Returns:
        The closure.
    """

    def _closure(message: str = "") -> None:
        """
        Prints the time elapsed since the previous call.

        Args:
            message: Custom message to print. The overall result will be: 'message: time_elapsed'.
        """
        nonlocal time_ref
        current_time = perf_counter()
        if message != "":
            print(f"{message}: {current_time - time_ref:.2f} seconds")
        time_ref = current_time

    return _closure


class Checkpoint:
    """
    Same thing as above but with a class.
    """

    def __init__(self, _time_reference: float = perf_counter()) -> None:
        """
        Initializes a new timer.

        Args:
             _time_reference: The time origin of the checkpoint. If omitted, will be set to the date of the call.
        """
        self._time_reference = _time_reference

    def __call__(self, message: str = "") -> None:
        """
        Prints the time elapsed since the previous call.

        Args:
            message: Custom message to print. The overall result will be: 'message: time_elapsed'.
        """
        current_time = perf_counter()
        if message != "":
            print(f"{message}: {current_time - self._time_reference}")
        self._time_reference = current_time
