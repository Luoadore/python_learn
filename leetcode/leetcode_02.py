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