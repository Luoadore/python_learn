# -*- coding:utf-8 -*-
"""
# 二维数组查找
class Solution:
    # array 二维列表
    def Find(self, target, array):
        # write code here
        if len(array[0]) != 0:
            for i in range(len(array)):
                if target <= array[i][-1]:
                    if target in array[i]:
                        return True
        return False
"""
"""
# 替换空格
class Solution:
    # s 源字符串
    def replaceSpace(self, s):
        # write code here
        if len(s) == None:
            return False
        else:
            old = len(s)
            black = 0
            for i in s:
                if i == ' ':
                    black += 1
            new = old + black * 3
            index = new - 1
            s_old = list(s)
            s_new = list(s)
            for i in range(black * 3):
                s_new.append(0)
            for i in range(old - 1, -1, -1):
                if s_old[i] == ' ':
                    s_new[index] = '0'
                    s_new[index - 1] = '2'
                    s_new[index - 2] = '%'
                    index = index - 2
                else:
                    s_new[index] = s_new[i]
            return "".join(s_new)
"""
"""
# 旋转数组的最小数字
class Solution:
    def minNumberInRotateArray(self, rotateArray):
        # write code here
        s = len(rotateArray)
        if s == 0:
            return 0
        else:
            begin = 0
            end = s - 1
            middle = 0
            while rotateArray[begin] >= rotateArray[end]:
                if end - begin == 1:
                    middle = end
                    break
                middle = (begin + end) // 2
                if rotateArray[middle] >= rotateArray[begin]:
                    if rotateArray[middle] == rotateArray[end]:
                        n_min = rotateArray[begin]
                        for i in range(begin, end):
                            if rotateArray[i] < n_min:
                                middle = i
                        break
                    else:
                        begin = middle
                else:
                    end = middle
            return rotateArray[middle]
"""
"""
# 从尾到头打印链表"""