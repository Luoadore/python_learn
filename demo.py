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
# 二进制中1的个数
# 按位与
class Solution:
    def NumberOf1(self, n):
        count = 0
        for j in range(32):
            i = 1
            i = i << j
            print(n & i)
            if n & i:
                count += 1
        return count
# 一个数减1后和本身与，得1的次数
    def NumberOf1(self, n):
        count = 0
        while n:
            count += 1
            n = (n - 1) & n
        return count
s = Solution()
print(s.NumberOf1(10))
"""
"""
# 数值的整数次方
class Solution:
    def Power(self, base, exponent):
        if base == 0:
            if exponent < 0:
                raise ValueError('Invalid error! ')
            else:
                return 0
        num, s = abs(exponent), 1
        for i in range(num):
            s = s * base
        if exponent < 0:
            s = 1.0 / s
        return s
# 递归方法
class Solution:
    def Power(self, base, exponent):
        if base == 0:
            if exponent < 0:
                raise ValueError('Invalid error! ')
            else:
                return 0
        if exponent >= 0:
            return self.power2(base, abs(exponent))
        else:
            return 1.0 / self.power2(base, abs(exponent))
    def power2(self, base, exponent):
        if exponent == 0:
            return 1
        if exponent == 1:
            return base
        s = self.power2(base, exponent >> 1)
        s = s * s
        if (exponent and 1) == 1:
            s = s * base
        return s
s = Solution()
s.Power(2,3)
"""
"""
# 调整数组顺序使奇数位于偶数前面
class Solution:
    def reOrderArray(self, array):
        re = []
        for i in range(len(array)):
            if array[i] % 2 != 0:
                re.append(array[i])
                array[i] = []
        for i in range(len(array)):
            if array[i] != []:
                re.append(array[i])
        return re
s = Solution()
print(s.reOrderArray([1, 2, 3, 4, 5, 6, 7]))
"""
"""
# 输出链表中倒数第k个节点
class ListNode:
    def __init__(self, x):
        self.x = x
        self.next = None
    def set_next(self, next):
        self.next = next
class Solution:
    def FindKthToTail(self, head, k):
        if head == None:
            return None
        if k == 0:
            return None
        i = head
        for index in range(k - 1):
            i = i.next
        if i == None:
            raise ValueError('Invalid k ! Out of the index.')
        else:
            j = head
            while i.next != None:
                i = i.next
                j = j.next
            return j
s = Solution()
a = ListNode(1)
b = ListNode(2)
c = ListNode(3)
d = ListNode(4)
e = ListNode(5)
a.set_next(b)
b.set_next(c)
c.set_next(d)
d.set_next(e)
print(s.FindKthToTail(a, 1).x)
"""
"""
# 反转链表
class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None
    def set_next(self, next):
        self.next = next
class Solution:
    # 返回ListNode
    def ReverseList(self, pHead):
        if pHead == None:
            return None
        elif pHead.next == None:
            return pHead

        head = pHead
        p_new = None
        while head != None:
            p_last = head.next
            head.next = p_new
            p_new = head
            head = p_last
        return p_new
a = ListNode(1)
b = ListNode(2)
c = ListNode(3)
d = ListNode(4)
e = ListNode(5)
a.set_next(b)
b.set_next(c)
c.set_next(d)
d.set_next(e)
s = Solution()
print(s.ReverseList(a).val)
print(a.val)
"""
"""
# 二叉树的镜像
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None
class Solution:
    # 返回镜像树的根节点
    def Mirror(self, root):
        if root == None:
            return root
        if not (root.left or root.right):
            return root
        temp = root.left
        root.left = root.right
        root.right = temp
        if root.left:
            self.Mirror(root.left)
        if root.right:
            self.Mirror(root.right)
        return root
"""
"""
# 顺时针打印矩阵
class Solution:
    # matrix类型为二维列表，需要返回列表
    def printMatrix(self, matrix):
        row = len(matrix)
        if row == 0:
            return []
        elif len(matrix[0]) == 0:
            return []
        else:
            col = len(matrix[0])
            start = 0
            array = []
            while (col > start * 2 and row > start * 2):
                array_min = self.printMatrixCircle(matrix, row, col, start)
                array.extend(array_min)
                start += 1
            return array
    def printMatrixCircle(self, matrix, row, col, start):
        array = []
        endx = col - 1 - start
        endy = row - 1 - start
        for i in range(start, endx + 1):
            array.append(matrix[start][i])
        if start < endy:
            for i in range(start + 1, endy + 1):
                array.append(matrix[i][endx])
        if ((start < endx) and (start < endy)):
            for i in range(-start - 2, -endx - 2, -1):
                array.append(matrix[endy][i])
        if ((start < endx) and (start < endy - 1)):
            for i in range(-start - 2, -endy - 1, -1):
                array.append(matrix[i][start])
        return array
s = Solution()
m = [[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]]
print(s.printMatrix(m))
"""