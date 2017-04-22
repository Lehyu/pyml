import time


def apply_sklearn_workaround():
    from functools import wraps
    from sklearn.base import BaseEstimator
    def safe_get_params(fn):
        @wraps(fn)
        def safe_wrapper(*args, **kwargs):
            result = None
            while True:
                try:
                    result = fn(*args, **kwargs)
                    break
                except IndexError as e:
                    # print(e, type(e))
                    time.sleep(1)
            return result

        return safe_wrapper
    BaseEstimator.get_params = safe_get_params(BaseEstimator.get_params)
