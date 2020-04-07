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

# 942 增减字符串匹配 简单
# 每次会把可以使用的数的集合中的最小值或最大值取出，并放到当前的位置
class Solution:
    def diStringMatch(self, S: str) -> List[int]:
        lo, hi = 0, len(S)
        res = []

        for x in S:
            if x == "I":
                res.append(lo)
                lo += 1
            if x == "D":
                res.append(hi)
                hi -= 1
        return res + [lo]

# 53 最大子序和 简单
# 动态规划
# 状态定义 数组dp
# 状态初始化
# 状态转移条件 dp[i] = max(dp[i - 1], 0) + nums[i]
class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        if len(nums) == 0:
            return 0
    
        sum_curr = res = nums[0]
        for i in range(1, len(nums)):
            sum_curr = max(sum_curr, 0) + nums[i]
            res = max(res, sum_curr)
        
        return res
# 分治法
# 定义基本情况
# 将问题分解为子问题并递归地解决它们
# 合并子问题的解获得原始问题的解
# TODO：目前没太看懂cross sum

# 169 多数元素 简单
# 哈希表（字典）
# O(n) O(n) 
class Solution:
    def majorityElement(self, nums: List[int]) -> int:
        counts = collections.Counter(nums)
        return max(counts.keys(), key=counts.get)

# 排序
# 如果将数组 nums 中的所有元素按照单调递增或单调递减的顺序排序，那么下标为「2/n」 的元素（下标从 0 开始）一定是众数。
# O(nlogn)（排序的空间复杂度） O(logn)
class Solution:
    def majorityElement(self, nums: List[int]) -> int:
        nums.sort()
        return nums[len(nums)//2]

# 分治
# 直到所有的子问题都是长度为 1 的数组。长度为 1 的子数组中唯一的数显然是众数，直接返回即可。
# 如果回溯后某区间的长度大于 1，我们必须将左右子区间的值合并。
# 如果它们的众数相同，那么显然这一段区间的众数是它们相同的值。
# 否则，我们需要比较两个众数在整个区间内出现的次数来决定该区间的众数。
class Solution:
    def majorityElement(self, nums: List[int]) -> int:
        def cross_major(left, right, lm, rm):
            if lm == rm:
                return lm
            
            lm_count = rm_count = 0
            for each in left + right:
                if each == lm:
                    lm_count += 1
                if each == rm:
                    rm_count += 1
            if lm_count > rm_count:
                return lm
            else:
                return rm
        
        def major(nums):
            if len(nums) == 1:
                return nums[0]

            mid = len(nums) // 2
            lm = major(nums[0: mid])
            rm = major(nums[mid:])
            maj = cross_major(nums[0: mid], nums[mid:], lm, rm)
            return maj
        
        return major(nums)

# 863 二叉树中所有距离为K的结点 中等
# DFS 深度优先搜索 + BFS 广度优先搜索
# 如果节点有指向父节点的引用，也就知道了距离该节点 1 距离的所有节点。
# 之后就可以从 target 节点开始进行深度优先搜索了。
class Solution:
    def distanceK(self, root: TreeNode, target: TreeNode, K: int) -> List[int]:
        def dfs(node, par=None):
            if node:
                node.par = par
                dfs(node.left, node)
                dfs(node.right, node)

        dfs(root)

        queue = collections.deque([(target, 0)])
        seen = {target}
        while queue:
            if queue[0][1] == K:
                return [node.val for node, d in queue]

            node, d = queue.popleft()
            for each in (node.left, node.right, node.par):
                if each and each not in seen:
                    seen.add(each)
                    queue.append((each, d+1))

        return []