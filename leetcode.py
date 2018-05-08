# coding: utf-8
# Luo ya'nan
# Remove duplicates from sorted array
# 
def remove_dup(nums):
    if len(nums) == 0:
        return None
    num, i = nums[0], 1
    while i < len(nums):
        while nums[i] == num:
            nums.remove(nums[i])
            print(nums)
            if len(nums) <= i:
                break
        if len(nums) > i:
            num = nums[i]
            i += 1
        else:
            break
    return len(nums) 

print(remove_dup([1,1]))
