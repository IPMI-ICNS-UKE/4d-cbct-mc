import functools
import inspect
import logging
import time
from typing import Any, Dict, Tuple

from ipmi.fused_types import Function

logger = logging.getLogger(__name__)


def _get_args_kwargs_with_values(
    func: Function, args: tuple, kwargs: dict, resolve_wrapped: bool = True
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    while True and resolve_wrapped:
        try:
            func = func.__wrapped__
        except AttributeError:
            break

    signature = inspect.signature(func)

    # positional only args, i.e. f(this, /, this_not)
    arg_params = [
        param
        for param in signature.parameters.values()
        if param.kind == param.POSITIONAL_ONLY
    ]
    kwarg_params = [
        param
        for param in signature.parameters.values()
        if param.kind in (param.POSITIONAL_OR_KEYWORD, param.KEYWORD_ONLY)
    ]

    # zip ends at shorter iterable, here: arg_params may be shorter than args
    args_dict = {param.name: arg_value for param, arg_value in zip(arg_params, args)}

    # kwargs passed as args, move to kwargs dict
    for i_kwarg, kwarg_value in enumerate(args[len(args_dict) :]):
        kwargs[kwarg_params[i_kwarg].name] = kwarg_value

    kwargs_dict = {
        param.name: kwargs.get(param.name, param.default) for param in kwarg_params
    }
    # join initially passed kwargs
    kwargs_dict.update(kwargs)

    return args_dict, kwargs_dict


def _as_parameterizable_decorator(decorator_func):
    @functools.wraps(decorator_func)
    def wrapper(*args, **kwargs):
        def parameterized_decorator(func):
            return decorator_func(func, *args, **kwargs)

        return parameterized_decorator

    return wrapper


@_as_parameterizable_decorator
def convert(
    func: Function, argument: str, converter: Function, convert_none: bool = False
):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        args_dict, kwargs_dict = _get_args_kwargs_with_values(func, args, kwargs)

        for d in (args_dict, kwargs_dict):
            if argument in d:
                arg = d[argument]
                if arg is not None or (arg is None and convert_none):
                    logger.debug(
                        f"Try to convert argument {argument!r} "
                        f"with value {d[argument]!r} "
                        f"using converter {converter!r}"
                    )
                    d[argument] = converter(d[argument])
                break
        else:  # no break in for loop above
            raise KeyError(f"Cannot find {argument} in args/kwargs")
        return func(*args_dict.values(), **kwargs_dict)

    return wrapper


@_as_parameterizable_decorator
def timing(func: Function, log_level: int = logging.DEBUG):
    @functools.wraps(func)
    def timing_wrapper(*args, **kwargs):
        t_start = time.monotonic()
        result = func(*args, **kwargs)
        t_end = time.monotonic()
        logger.log(
            level=log_level, msg=f"{func.__name__} took {t_end - t_start} seconds"
        )
        return result

    return timing_wrapper
