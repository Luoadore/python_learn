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
