from contextlib import contextmanager
import logging


def add_async_graph(module, log_flops: bool = False):
    module.asy_graph = None
    module.asy_flops_log = [] if log_flops else None
    return module


def make_asynchronous(module, initialization_func, processing_func):
    module.sync_forward = module.forward
    def async_forward(*args, **kwargs):
        with async_context(module, initialization_func, processing_func) as func:
            output = func(module, *args, **kwargs)
        return output
    module.forward = async_forward
    return module


@contextmanager
def async_context(module, initialization_func, processing_func):
    if module.asy_graph is None:
        logging.debug(f"Graph initialization of module {module}")
        yield initialization_func
    else:
        logging.debug(f"Calling processing of module {module}")
        yield processing_func
