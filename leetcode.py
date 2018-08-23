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

# 二叉树的中序遍历
# 迭代的方法
# 思路：从根节点开始，先将根节点压入栈，然后再将其所有左子结点压入栈，
# 然后取出栈顶节点，保存节点值，再将当前指针移到其右子节点上，
# 若存在右子节点，则在下次循环时又可将其所有左子结点压入栈中。
def inorderTraversal(self, root):
    stack = []
    l = []
    node = root
    while len(stack) != 0 or node:
        if node:
            stack.append(node)
            node = node.left
        else:
            node = stack.pop()
            l.append(node.val)
            node = node.right
    return l