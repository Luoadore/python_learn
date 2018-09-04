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

# 岛屿的数量
# DFS 
# 非递归
def num_islands(grid):
    count = 0

    for i in range(len(grid)):
        for j in range(len(grid[0])):

            if grid[i][j] == '1':
                count += 1

                # stack dfs
                stack = [(i, j)]

                while len(stack) != 0:
                    i0, j0 = stack.pop()

                    if (i0 < 0 or i0 >= len(grid)) or \
                       (j0 < 0 or j0 >= len(grid[0])) or \
                       grid[i0][j0] == '0':
                        continue

                    grid[i0][j0] = '0'

                    # add all 4neighbors
                    stack.append((i0 - 1, j0))
                    stack.append((i0, j0 + 1))
                    stack.append((i0 + 1, j0))
                    stack.append((i0, j0 - 1))
    return count

# 01矩阵
# 找出每个元素到最近的 0 的距离。
# BFS，距离
def updateMatrix(matrix):
    import numpy as np
    import sys
    dist = np.zeros([len(matrix), len(matrix[0])])
    queue = []
    # 初始化: distance for each 0 cell is 0 and distance for each 1 is INT_MAX
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            if matrix[i][j] == 0:
                queue.append((i, j))
            else:
                dist[i][j] = sys.maxsize

    # update & add new to queue
    # 每次广度层次遍历上下左右四个节点
    neighbor4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    while len(queue) != 0:
        i0, j0 = queue.pop(0)
        for r, c in neighbor4:
            r0, c0 = i0 + r, j0 + c
            if r0 < len(matrix) and r0 >= 0 and c0 < len(matrix[0]) and c0 >= 0:
                if dist[i0][j0] + 1 < dist[r0][c0]:
                    dist[r0][c0] = dist[i0][j0] + 1
                    queue.append((r0, c0))

    return dist

# print(updateMatrix([[0,0,0],[0,1,0],[1,1,1]]))
 
# 二叉树展开为链表
# 原地：寻找右孩子的前序
# 把左孩子变为右孩子，如果有右孩子，最后一个右孩子指向右子树
def flatten(root):
	while root:
		if root.left:
			pre = root.left
			while pre.right:
				pre = pre.right
			pre.right = root.right
			root.right = root.left
			root.left = None
		root = root.right

# 454四数相加
# 思路：把四个数组分为两组，每组包含两个数组。
# 其中一组中的任意两个值和存入hashmap中，然后在hashmap查找另外两个数组的值的组合。
# 相当于转化为了一个two sum问题
def four_sum_count(A, B, C, D):
	AB = collections.Counter(a + b for a in A for b in B)
	return sum(AB[-c-d] for c in C for d in D)

def f_s_c(A, B, C, D):
	# 会超时
	AB = {}
	for i in range(len(A)):
		for j in range(len(B)):
			sum1 = A[i] + B[j]
			# 以下四句可以用AB[sum1] = AB.get(sum1, 0) + 1
			if sum1 not in AB.keys():
				AB[sum1] = 1
			else:
				AB[sum1] += 1

	count = 0
	for i in range(len(C)):
		for j in range(len(D)):
			sum2 = C[i] + D[j]
			if -sum2 in AB.keys():
				count += AB[-sum2]
	return count

# 797所有可能的路径
# 有向无环图