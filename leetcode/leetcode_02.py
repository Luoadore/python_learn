# coding: utf-8
# 2020-04-01

# 344 反转字符 简单
class Solution:
    def reverseString(self, s: List[str]) -> None:
        """
        Do not return anything, modify s in-place instead.
        """
        s.reverse()

    # 双指针方法
    def reverseString(self, s):
        left, right = 0, len(s) - 1
        while left < right:
            s[left], s[right] = s[right], s[left]
            left, right = left + 1, right - 1

# 415 字符串相加 简单
# 不可以转换为int型做
class Solution:
    def addStrings(self, num1: str, num2: str) -> str:
        l1, l2 = len(num1) - 1, len(num2) - 1
        while l1 < l2:
            num1 = '0' + num1
            l1 += 1
        while l1 > l2:
            num2 = '0' + num2
            l2 += 1
        str_re, carry = '0', 0
        print(type(str_re))
        for i in range(len(num1)):
            sum_ = int(num1[i]) + int(num2[i])
            str_re += str(sum_ % 10 + carry)
            carry = sum_ / 10
        str_re += str(carry)
        return str_re

# 11 盛最多水的容器 中等
# 双指针，受限于最小长度和两数之间的距离，则比较大小并将较短指针向中间移动即可
class Solution:
    def maxArea(self, height: List[int]) -> int:
        if len(height) <= 1:
            return len(height) * height
        left, right = 0, len(height) - 1
        maxarea = 0
        while left != right:
            min_height = min(height[left], height[right])
            maxarea = max(maxarea,  min_height * (right - left))
            if height[left] == min_height:
                left += 1
            else:
                right -= 1
        return maxarea

# 263 丑数 简单
# 不断取余直到//结果为1
class Solution:
    def isUgly(self, num: int) -> bool:
        if num == 0: return False
        if num == 1:return True
        if num % 2 == 0: return self.isUgly(num // 2)
        if num % 3 == 0: return self.isUgly(num // 3)
        if num % 5 == 0: return self.isUgly(num // 5)
        return False

# 264 丑数2 中等
# 堆 存放预先计算好的丑数放到数组中，哈希表跟踪堆中的元素避免重复，每次弹出最小数字并添加到数组，再计算2、3、5的倍数
# 时间复杂度O(1)
from heapq import heappop, heappush
class Ugly:
    def __init__(self):
        seen = {1, }
        self.nums = nums = []
        heap = []
        heappush(heap, 1)

        for _ in range(1690):
            curr_ugly = heappop(heap)
            nums.append(curr_ugly)
            for i in [2, 3, 5]:
                new_ugly = curr_ugly * i
                if new_ugly not in seen:
                    seen.add(new_ugly)
                    heappush(heap, new_ugly)
class Solution:
    u = Ugly()
    def nthUglyNumber(self, n):
        return self.u.nums[n-1]

# 动态规划
# 一个预存数组和三个指针，三个指针指向的位置分别是需要[2，3，5]做乘法的位置
class Ugly:
    def __init__(self):
        self.nums = nums = [1,]
        i2 = i3 = i5 = 0

        for i in range(1, 1690):
            ugly = min(nums[i2] * 2, nums[i3] * 3, nums[i5] * 5)
            nums.append(ugly)

            if ugly == nums[i2] * 2:
                i2 += 1
            if ugly == nums[i3] * 3:
                i3 += 1
            if ugly == nums[i5] * 5:
                i5 += 1

class Solution:
    u = Ugly()
    def nthUglyNumber(self, n: int) -> int:
        return self.u.nums[n-1]

# 1201 丑数3 中等
# 解题思路 二分法+最小公倍数
# 利用最小公倍数可以计算[0, X]间有多少丑数，之后二分查找即可；前者分为7种情况:容斥原理，后者注意计算二分查找左边界值
# 容斥原理：sum(情况) = X/a + X/b + X/c - X/LCM_a_b - X/LCM_a_c - X/LCM_b_c + X/LCM_a_b_c
# 左边界值：K - min(K%a,K%b,K%c) = X
# https://leetcode-cn.com/problems/ugly-number-iii
class Solution:
    def nthUglyNumber(self, n: int, a: int, b: int, c: int) -> int:
        # LCM is Least Common Multiple
        def LCM(a, b):
            multi = a * b
            while b > 0:
                a, b = b, a % b
            return multi / a
        
        # inclusion-exclusion principle
        def uglynums(x, a, b, c):
            LCM_a_b = LCM(a, b)
            LCM_a_c = LCM(a, c)
            LCM_b_c = LCM(b, c)
            LCM_a_b_c = LCM(LCM_a_b, c)
            res = x//a + x//b + x//c - x//LCM_a_b - x//LCM_a_c - x//LCM_b_c + x//LCM_a_b_c
            return int(res)

        # binary search
        def bsearch(left, right, n, a, b, c):
            if left >= right:
                return left
            mid = (left + right) / 2
            nums = uglynums(mid, a, b, c)
            if nums == n:
                return mid
            if nums >= n:
                return bsearch(left, mid - 1, n, a, b, c)
            return bsearch(mid + 1, right, n, a, b, c)
        
        # main
        low = min(a, b, c)
        high = low * n

        res = bsearch(low, high, n, a, b, c)

        res = res - min(res%a, res%b, res%c)
        return int(res)

# 100 相同的树 简单

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
# 递归，检查每一个节点是否为空，然后判断值，再递归子节点
class Solution:
    def isSameTree(self, p: TreeNode, q: TreeNode) -> bool:
        if not p and not q:
            return True
        if not p or not q:
            return False
        if p.val != q.val:
            return False
        return self.isSameTree(p.left, q.left) and self.isSameTree(p.right, q.right)

# 双向队列，弹出相应的节点，判断
from collections import deque
class Solution:
    def isSameTree(self, p: TreeNode, q: TreeNode) -> bool:

        def check(p, q):
            if not p and not q:
                return True
            if not q or not p:
                return False
            if p.val != q.val:
                return False
            return True
        
        deq = deque([(p, q),])
        while deq:
            p, q = deq.popleft()
            if not check(p, q):
                return False
            if p:
                deq.append((p.right, q.right))
                deq.append((p.left, q.left))
        return True

# 102 二叉树的层序遍历 中等

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

# BFS 广度优先遍历 将树上顶点按照层次依次放入队列结构， level和deque
class Solution:
    def levelOrder(self, root: TreeNode) -> List[List[int]]:
    	"""
        :type root: TreeNode
        :rtype: List[List[int]]
        """
        levels = []
        if not root:
            return levels

        level = 0
        queue = deque([root,])
        while queue:
            # start the current level
            levels.append([])
            # number of elements in the current level
            level_length = len(queue)

            for i in range(level_length):
                node = queue.popleft()
                # fullfill the current level
                levels[level].append(node.val)

                # add child nodes of the current level in the queue for the next level
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)

            # go to next level
            level += 1

        return levels

# DFE 深度优先遍历 递归函数，参数是当前节点和节点的层次
class Solution:
    def levelOrder(self, root: TreeNode) -> List[List[int]]:
        """
        :type root: TreeNode
        :rtype: List[List[int]]
        """
        levels = []
        if not root:
            return levels

        def helper(node, level):
            # start the current level
            if len(levels) == level:
                levels.append([])

            # append the current node value
            levels[level].append(node.val)

            # process child nodes for the next level
            if node.left:
                helper(node.left, level + 1)
            if node.right:
                helper(node.right, level + 1)

        helper(root, 0)
        return levels

