# coding: utf-8
# Luo ya'nan

# Remove duplicates from sorted array
# O(n) O(1)
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

# Remove duplicates from sorted array 2
# O(n) O(1)
def remove_dup(nums):
    if len(nums) <= 2: 
        return len(nums)
    index = 2
    for i in range(2, len(nums)):
        if nums[index - 2] != nums[i]:
            nums[index] = nums[i]
            index += 1
    return index

print(remove_dup([1,1,1,2,3,4]))
