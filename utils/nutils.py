'''
handle number
'''
import numpy as np

def is_int(num):
    if num < 0:
        return False
    num = str(num)
    nums = num.split('.')
    return len(nums) == 1 or (float(nums[1]) == 0)

def shannon_ent(prob):
    if prob == 0:
        return 0
    else:
        return prob*np.log2(prob)
def check_discrete(values):
    discrete = True
    for val in values:
        discrete &= is_int(val)
        if not discrete:
            break
    return discrete

