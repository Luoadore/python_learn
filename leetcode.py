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
# Let N be the number of nodes in the graph. 
# If we are at node N-1, the answer is just the path {N-1}. Otherwise, 
# if we are at node node, the answer is {node} + {path from nei to N-1} for each neighbor nei of node. 
# This is a natural setting to use a recursion to form the answer.
def allPathsSourceTarget(graph):
	N = len(graph)

	def solver(node):
		if node == N - 1:
			return [[N - 1]]
		ans = []
		for nei in graph[node]:
			for path in solver(nei):
				ans.append([node] + path)
		return ans
	return solver(0)

# 763划分字母区间
# 贪心，找到每个字母最后出现的位置
# 维护partition的起点和终点
# 时间复杂度O(n)
def partitionLabels(S):
	last = {c: i for i, c in enumerate(S)}
	print(last)
	j = anchor = 0
	ans = []
	for i, c in enumerate(S):
		j = max(j, last[c])
		if i == j:
			ans.append(i - anchor + 1)
			anchor = i + 1
	return ans

#print(partitionLabels('ababcabde'))

# 56合并区间
# 
def merge(intervals):
	inter = [x[i] for x in intervals for i in range(2)]
	print(inter)
	i = 1
	while i < len(inter) - 1:
		j = i + 1
		if j < len(inter) and inter[i] >= inter[j]:
			inter.pop(i)
			inter.pop(j - 1)
		i += 2
	print(inter)
	return [[inter[i], inter[j]] for i, j in zip(range(0, len(inter) - 1, 2), range(1, len(inter), 2))]


print(merge([[1,3],[2,6],[8,10],[15,18]]))
print(merge([[1,4],[4,5]]))

# 13罗马数字转整数
def romanToInt(s):
    """
    type s: str
    rtype: int
    """
    roman_dict = {'I':1, 'V':5, 'X':10, 'L':50, 'C':100, 'D':500, 'M':1000}
    s_len = len(s)
    integer, i, addend, flag = 0, 0, 0, True
    while i < s_len:
        if i + 1 < s_len:
            print('nice:', i)
            if s[i] == 'I' and (s[i + 1] == 'V' or s[i + 1] == 'X'):
                addend = roman_dict[s[i + 1]] - roman_dict[s[i]]
                i += 2
                flag = False
            elif s[i] == 'X' and (s[i + 1] == 'L' or s[i + 1] == 'C'):
                addend = roman_dict[s[i + 1]] - roman_dict[s[i]]
                i += 2
                flag = False
            elif s[i] == 'C' and (s[i + 1] == 'D' or s[i + 1] == 'M'):
                addend = roman_dict[s[i + 1]] - roman_dict[s[i]]
                i += 2
                flag = False
        if flag:
            addend = roman_dict[s[i]]
            i += 1
        #print(addend)
        integer += addend       
    return integer

print(romanToInt('MCMXCIV'))
print(romanToInt('LVIII'))
print(romanToInt('IX'))
print(romanToInt('IV'))
print(romanToInt('III'))

# 42 接雨水
# 分治：最大次大之间区域间雨水量
class Solution(object):
    def trap(self, height):
        """
        :type height: List[int]
        :rtype: int
        """
        def raindrop(height):
            rain = 0
            if len(height) < 2:
                return rain
            h = min(height[0], height[-1])
            rain = h * (len(height) - 2) - sum(height[1:-1])
            return rain
        rain = 0
        l, r = 0, len(height) - 1
        if l < r - 1: 
            i1, i2 = height.index(max(height)), 0
            minus = sys.maxsize
            for j in range(len(height)):
                if j != i1 and height[i1] - height[j] < minus:
                    minus = height[i1] - height[j]
                    i2 = j
            l, r = min(i1, i2), max(i1, i2)
            rain = raindrop(height[l:r + 1]) + self.trap(height[0:l + 1]) + self.trap(height[r:])
        return rain

# 84 柱状图中最大的矩形
# 分治：最小值对应的面积，有序数列优化（注意时长）
class Solution(object):
    def largestRectangleArea(self, heights):
        """
        :type heights: List[int]
        :rtype: int
        """
        def area(heights):
            min_area = 0
            if len(heights) >= 1:
                min_heights = min(heights)
                min_index = heights.index(min_heights)
                min_area = min_heights * len(heights)
                
                max_tmp = 0
                if heights == sorted(heights):
                    for h in range(len(heights)):
                        tmp = (len(heights) - h) * heights[h]
                        if max_tmp < tmp:
                            max_tmp = tmp
                            
                    return max_tmp
                
                if heights == sorted(heights)[::-1]:
                    for h in range(len(heights)):
                        tmp = (h + 1) * heights[h]
                        if max_tmp < tmp:
                            max_tmp = tmp
                            
                    return max_tmp
                
                return max(min_area, area(heights[:min_index]), area(heights[min_index+1:]))
            else:
                return min_area
        
        area_heights = area(heights)
        return area_heights

# 85 最大矩形
# 先求每行连续1的个数，再按列算柱状图最大矩形
class Solution(object):
    def maximalRectangle(self, matrix):
        """
        :type matrix: List[List[str]]
        :rtype: int
        """
        def get_bar(heights):
            
            if len(heights) == 0:
                return []
            
            if heights[0] == "1":
                flag = 1
            else:
                flag = 0
            heights[0] = int(heights[0])
            for i in range(1, len(heights)):
                if heights[i] == "1":
                    if flag == 1:
                        heights[i] = heights[i-1] + 1
                    else:
                        heights[i] = 1
                        flag = 1
                else:
                    heights[i] = 0
                    flag = 0

            return heights

        def max_rectangle(heights):

            def area(heights):
                min_area = 0
                if len(heights) >= 1:
                    min_heights = min(heights)
                    min_index = heights.index(min_heights)
                    min_area = min_heights * len(heights)
                    
                    max_tmp = 0
                    if heights == sorted(heights):
                        for h in range(len(heights)):
                            tmp = (len(heights) - h) * heights[h]
                            if max_tmp < tmp:
                                max_tmp = tmp
                                
                        return max_tmp
                    
                    if heights == sorted(heights)[::-1]:
                        for h in range(len(heights)):
                            tmp = (h + 1) * heights[h]
                            if max_tmp < tmp:
                                max_tmp = tmp
                                
                        return max_tmp
                    
                    return max(min_area, area(heights[:min_index]), area(heights[min_index+1:]))
                else:
                    return min_area
        
            return area(heights)
        
        if len(matrix) == 0:
            return 0

        max_area = 0

        matrix_bar = []
        for each in matrix:
            each_bar = get_bar(each)
            matrix_bar.append(each_bar)
        print(matrix_bar)
        
        for i in range(len(matrix[0])):
            each_bar = [matrix[j][i] for j in range(len(matrix))]
            each_area = max_rectangle(each_bar)
            if each_area > max_area:
                max_area = each_area
        
        return max_area

# 141 环形链表
# 快慢双指针 时间复杂度O(n),空间复杂度O(1)

# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def hasCycle(self, head: ListNode) -> bool:
        if head == None or head.next == None:
            return False
        fast, slow = head.next, head
        while fast != slow:
            if fast == None or fast.next == None:
                return False
            fast = fast.next.next
            slow = slow.next
        return True
