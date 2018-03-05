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
# 从尾到头打印链表
class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None

    def set_attr(self, next):
        self.next = next

class Solution:
    def printListFromTailToHead(self, listNode):
        l_head = listNode
        if l_head == None:
            return False
        else:
            nodelist = []
            nodelist_reverse = []
            while listNode.next != None:
                nodelist.append(listNode.val)
                listNode = listNode.next
            nodelist.append(listNode.val)
            for i in range(len(nodelist)):
                nodelist_reverse.append(nodelist.pop())
            return nodelist_reverse

a = ListNode(1)
b = ListNode(2)
c = ListNode(3)
a.set_attr(b)
b.set_attr(c)
c.set_attr(None)
s = Solution()
print(s.printListFromTailToHead(a))
"""
"""
# 重建二叉树
# 前序遍历：NLR, 中序遍历：LNR，后序遍历：LRN
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

    def set_kids(self, l, r):
        self.left = l
        self.right = r

class Solution:
    def reConstructBinaryTree(self, pre, tin):
        # 递归
        if not pre or not tin:
            return None
        root = TreeNode(pre.pop(0))
        index = tin.index(root.val)
        root.left = self.reConstructBinaryTree(pre, tin[: index])
        root.right = self.reConstructBinaryTree(pre, tin[index + 1 :])
        return root

pre = [1, 2, 4, 7, 3, 5, 6, 8]
tin = [4, 7, 2, 1, 5, 3, 8, 6]
s = Solution()
print(s.reConstructBinaryTree(pre, tin).right.val)
"""
"""
# 用两个栈实现队列
# 栈先进先出，队列先进后出
class Queue():
    def __init__(self):
        self.stack1 = []
        self.stack2 = []
    def push(self, node):
        self.stack1.append(node)
    def pop(self):
        if len(self.stack2) == 0:
            while len(self.stack1) != 0:
                self.stack2.append(self.stack1.pop())
        return self.stack2.pop()
s = Queue()
s.push(1)
s.push(2)
s.push(3)
print(s.stack1)
s.pop()
print(s.stack2)
"""
"""
# 佩波那契数列
class Solution:
    def Fibonacci(self, n):
        a = [0, 1]
        if n > 1:
            for i in range(2, n + 1):
                a.append(a[i - 2] + a[i - 1])
        return a[n]
"""
"""
# 跳台阶
# 台阶1，1种；台阶2，2种；台阶n，跳1次之后f（n-1）种，跳2个之后f（n-2）种，求和。
class Solution:
    def jumpFloor(self, number):
        n = [1, 2]
        if number > 2:
            for i in range(2, number + 1):
                n.append(n[i - 2] + n[i - 1])
        if number < 1:
            return False
        else:
            return n[number - 1]
"""
"""
# 变态跳台阶"
# f(n) = f(n-1) + f(n-2) +...+ f(1)，即f(n) = 2f(n-1)
import math
class Solution:
    def jumpFloorII(self, number):
        n = [1]
        if number > 0:
            #for i in range(1, number + 1):
             #   n_0 = 0
              #  for j in range(i):
               #     n_0 += n[j]
                #n.append(n_0 + 1)
        #return n[number - 1]
            return math.pow(2, number - 1)
"""
"""
# 矩形覆盖
# f(n) = f(n - 1) + f(n - 2), 一个2*1竖、横放时
class Solution:
    def rectCover(self, number):
        n = [1, 2]
        if number > 2:
            for i in range(2, number + 1):
                n.append(n[i - 1] + n[i - 2])
        return n[number - 1]
"""
"""
# 二进制中1的个数"""
