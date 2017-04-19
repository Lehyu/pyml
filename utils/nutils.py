'''
handle number
'''
import numpy as np
import random
import itertools


def is_int(num):
    if num < 0:
        return False
    num = str(num)
    nums = num.split('.')
    return len(nums) == 1 or (float(nums[1]) == 0)


def generate_combinations(cols, min_num):
    """
    :param cols: n tuple
    :return:
    """
    min_num = 0 if min_num < 0 else min_num
    min_num = min_num if min_num < len(cols) else len(cols)
    combs = set()
    for col in itertools.product(*cols):
        for r in range(min_num, len(col) + 1):
            for comb in itertools.combinations(col, r):
                combs.add(comb)
    combs = sorted(list(combs), key=lambda item: len(item))
    for comb in combs:
        yield comb


def shannon_ent(prob):
    if prob == 0:
        return 0
    else:
        return prob * np.log2(prob)


def check_discrete(values):
    discrete = True
    for val in values:
        discrete &= is_int(val)
        if not discrete:
            break
    return discrete


def shuffle(n_samples):
    indexes = list(range(n_samples))
    res = []
    while indexes:
        ix = random.randrange(0, len(indexes))
        res.append(indexes[ix])
        indexes.pop(ix)
    return res


def batch(n_samples, batch_size):
    res = shuffle(n_samples)
    iters = int(np.ceil(n_samples / batch_size))
    for iter in range(iters):
        start = iter * batch_size
        if iter == iters - 1:
            yield res[start:]
        else:
            yield res[start: start + batch_size]


def clip_by_value(value, v_min, v_max):
    return value.clip(v_min, v_max)


def sigmoid(a):
    return 1 / (1 + np.exp(-a))


def sofmax(a):
    tmp = np.exp(a)
    return tmp / np.sum(tmp, axis=1).reshape((-1, 1))


def corr(X, y):
    return np.dot(y, X) / np.sqrt(np.dot(y ** 2, X ** 2))
