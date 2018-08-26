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
def inorderTraversal(root):
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
    
# Maximal Square
# DP
# dp(i, j)=min(dp(i−1, j), dp(i−1, j−1), dp(i, j−1))+1.
# 找到maxqlens
def maximalSquare(matrix):
    rows = len(matrix)
    cols = len(matrix[0])
    dp = [[] for _ in range(rows)]
    for i in range(rows):
        for _ in range(cols):
            dp[i].append(0)  
    for row in range(rows):
        dp[row][0] = matrix[row][0]
    for col in range(cols):
        dp[0][col] = matrix[0][col]
    for row in range(1, rows):
        for cols in range(1, cols):
            dp[row][col] = min(dp[row-1, col-1], dp[row, col-1], dp[row-1, col]) + 1
    maxqlen = max(max(dp))
    return maxqlen * maxqlen

# 跳跃游戏
# 贪心算法
# 维护一个变量reach，表示最远能到达的位置
def can_jump(lists):
	reach = 0
	for i in range(len(lists)):
		if i > reach or reach >= len(lists) - 1:
			break
		reach = max(reach, i + lists[i])
	return reach >= len(lists) - 1

# 树的最大深度
# 递归
def max_depth(root):
    if root == None:
        return 0
    return max(max_depth(root.right), max_depth(root.left)) + 1

# 非递归
def max_depth(root):
    if root == None:
        return 0
    queue = []
    queue.append(root)
    height = 0
    layer_num = len(queue)
    while len(queue) != 0:
        node = queue.pop(0)
        layer_num -= 1
        if node.left:
            queue.append(node.left)
        if node.right:
            queue.append(node.right)
        if layer_num == 0:
            height += 1
            layer_num = len(queue)
    return height