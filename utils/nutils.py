'''
handle number
'''

def is_int(num):
    if num < 0:
        return False
    num = str(num)
    nums = num.split('.')
    return len(nums) == 1 or (float(nums[1]) == 0)