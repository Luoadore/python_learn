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
# 合并两个排序的链表
class Solution:
    # 返回合并后列表
    def Merge(self, pHead1, pHead2):
        if pHead1 == None:
            return pHead2
        if pHead2 == None:
            return pHead1
        if pHead1.val < pHead2.val:
            p_new = pHead1
            p_new.next = self.Merge(pHead1.next, pHead2)
        else:
            p_new = pHead2
            p_new.next = self.Merge(pHead1, pHead2.next)
        return p_new
"""
"""
# 树的子结构
class Solution:
    def HasSubtree(self, pRoot1, pRoot2):
        if pRoot1 == None:
            return False
        if pRoot2 == None:
            return False
        result = False
        if pRoot1.val == pRoot2.val:
            result = self.DoesTree1HaveTree2(pRoot1, pRoot2)
        if result != True:
            result = self.HasSubtree(pRoot1.left, pRoot2)
        if result != True:
            result = self.HasSubtree(pRoot1.right, pRoot2)
        return result
    def DoesTree1HaveTree2(self, pRoot1, pRoot2):
        # tree2没遍历完，tree1遍历完
        if (pRoot1 == None) and (pRoot2 != None):
            return False
        # tree2遍历完
        if pRoot2 == None:
            return True
        if pRoot1.val != pRoot2.val:
            return False
        result = (self.DoesTree1HaveTree2(pRoot1.left, pRoot2.left) and self.DoesTree1HaveTree2(pRoot1.right, pRoot2.right))
        return result
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
"""
# 包含min函数的栈
# 两个栈，一个存数据，一个存最小值
class Solution:
    def __init__(self):
        self.m_data = []
        self.m_min = []
    def push(self, node):
        self.m_data.append(node)
        if len(self.m_min) == 0:
            self.m_min.append(node)
        elif node <= self.m_min[-1]:
            self.m_min.append(node)
        else:
            self.m_min.append(self.m_min[-1])
    def pop(self):
        if (len(self.m_data) != 0) and (len(self.m_min) != 0):
            self.m_data.pop()
            self.m_min.pop()
    def top(self):
        if (len(self.m_data) != None) and (len(self.m_min) != 0):
            return self.m_data[-1]
    def min(self):
        if (len(self.m_data) != None) and (len(self.m_min) != 0):
            return self.m_min[-1]
"""

"""
# 栈的压入、弹出序列
# 辅助栈，比较pop栈顶和从push取到辅助栈栈顶是否相同
class Solution:
    def IsPopOrder(self, pushV, popV):
        if (len(pushV) == 0) or (len(popV) == 0) or (len(pushV) != len(popV)):
            return False
        judge = []
        for i in pushV:
            judge.append(i)
            while len(judge) and (judge[-1] == popV[0]):
                judge.pop()
                popV.pop(0)
        if len(judge):
            return False
        return True
s = Solution()
print(s.IsPopOrder([1, 2, 3, 4, 5], [4, 5, 3, 2, 1]))
print(s.IsPopOrder([1, 2, 3, 4, 5], [4, 3, 5, 1, 2]))
"""

"""
# 从上往下打印二叉树
# 广度优先遍历用队列实现
class Solution:
    # 返回从上到下每个节点值列表，例：[1,2,3]
    def PrintFromTopToBottom(self, root):
        tree = []
        tree_queue = []
        if root != None:
            tree_queue.append(root)
            while len(tree_queue) != 0:
                node = tree_queue.pop(0)
                tree.append(node.val)
                if node.left != None:
                    tree_queue.append(node.left)
                if node.right != None:
                    tree_queue.append(node.right)
        return tree
"""

"""
# 二叉搜索树的后序遍历序列
class Solution:
    def VerifySquenceOfBST(self, sequence):
        if len(sequence) == 0:
            return False
        root = sequence[-1]
        index = 0
        left, right = [], []
        while sequence[index] < root:
            left.append(sequence[index])
            index += 1
        for i in range(index, len(sequence) - 1):
            right.append(sequence[i])
        right = [x for x in right if x > root]
        result = False
        if len(left) + len(right) == len(sequence) - 1:
            result = True
        if len(left) != 0:
            result = result and self.VerifySquenceOfBST(left)
        if len(right) != 0:
            result = result and self.VerifySquenceOfBST(right)
        return result
s = Solution()
print(s.VerifySquenceOfBST([5, 7, 6, 9, 11, 10, 8]))
print(s.VerifySquenceOfBST([4, 6, 7, 5]))
print(s.VerifySquenceOfBST([7, 4, 6, 5]))
"""

"""
# 二叉树中和为某一值的路径
# 深度优先遍历用栈实现
# not so good at this Q
class Solution:
    # 返回二维列表，内部每个列表表示找到的路径
    def FindPath(self, root, expectNumber):
        if root == None:
            return []
        if root.left == None and root.right == None and root.val == expectNumber:
            return [[root.val]]
        res = []
        res_left = self.FindPath(root.left, expectNumber - root.val)
        res_right = self.FindPath(root.right, expectNumber - root.val)
        for i in res_left + res_right:
            res.append([root.val] + i)
        return res
"""

"""
# 复杂链表的复制
# 分治思想
class RandomListNode:
     def __init__(self, x):
         self.label = x
         self.next = None
         self.random = None
class Solution:
    # 返回 RandomListNode
    def Clone(self, pHead):
        if pHead == None:
            return pHead
        self.Clonenext(pHead)
        self.Clonerandom(pHead)
        return self.Splitnew(pHead)
    def Clonenext(self, pHead):
        # 克隆每个节点位于原节点之后
        old = pHead
        while old != None:
            # 这种方法有可能出现none type
            # new = old
            # new_next = old.next
            # old.next = new
            # new.next = new_next
            # old = new_next
            new = RandomListNode(old.label)
            new.next = old.next
            old.next = new
            old = new.next
        return pHead
    def Clonerandom(self, pHead):
        # 克隆随机指针
        node = pHead
        while node != None:
            if node.random != None:
                ran = node.random
                node.next.random = ran.next
            node = node.next.next
        return pHead
    def Splitnew(self, pHead):
        cur = pHead
        res = pHead.next
        while cur.next != None:
            temp = cur.next
            cur.next = temp.next
            cur = temp
        return res
"""

"""
# 二叉搜索树与双向链表
def Convert(self, pRootOfTree):
        if pRootOfTree == None:
            return pRootOfTree
        if pRootOfTree.left == None and pRootOfTree.right == None:
            return pRootOfTree
        # 处理左子树
        self.Convert(pRootOfTree.left)
        left = pRootOfTree.left

        # 将左子树链表中最后一个值与根相连
        if left:
            while left.right:
                left = left.right
            pRootOfTree.left, left.right = left, pRootOfTree

        # 处理右子树
        self.Convert(pRootOfTree.right)
        right = pRootOfTree.right

        # 将右子树最前一个值与根相连
        if right:
            while right.left:
                right = right.left
            pRootOfTree.right, right.left = right, pRootOfTree

        while pRootOfTree.left:
            pRootOfTree = pRootOfTree.left
        return pRootOfTree
"""
"""
# 字符串的排列
import itertools
class Solution:
    def Permutation(self, ss):
        if len(ss) == 0:
            return []
        #l = sorted(list(set(map(''.join, itertools.permutations(ss)))))
        l = sorted(list(map(''.join, itertools.combinations(ss, 2))))
        return l
s = Solution()
print(s.Permutation('cab'))
"""

"""
# 数组中出现次数超过一半的数
class Solution:
    def MoreThanHalfNum_Solution(self, numbers):
        if not numbers:
            return 0
        num = set(numbers)
        print(num)
        if (len(num) - 1) <= len(numbers) / 2:
            for i in num:
                var = [x for x in numbers if x == i]
                print(var)
                if len(var) > len(numbers) / 2:
                    return i
        return 0
        # solution 2
        import collections
        num = collections.Count(numbers)
        for k, v in num:
            if v > len(numbers) / 2:
                return k
        return 0
s = Solution()
print(s.MoreThanHalfNum_Solution([1,2,3,2,4,2,5,4,2]))
"""

"""
# 最小的k个数
class Solution:
    def GetLeastNumbers_Solution(self, tinput, k):
        if not tinput or not k or k > len(tinput):
            return []
        t = sorted(tinput)
        return t[: k]
s = Solution()
print(s.GetLeastNumbers_Solution([2,4,6,3,7,5], 3))
"""

"""
# 连续子数组的最大和
class Solution:
    def FindGreatestSumOfSubArray(self, array):
        if len(array) == 0:
            return False
        if len(array) == 1:
            return array[0]
        sum = 0
        max_sum = -0xffffff
        for i in array:
            sum += i
            if sum > max_sum:
                max_sum = sum
            if sum < 0:
                sum = 0
        return max_sum
s = Solution()
print(s.FindGreatestSumOfSubArray([6,-3,-2,7,-15,1,2,2]))
print(s.FindGreatestSumOfSubArray([-2,-8,-1,-6]))
"""

"""
# 整数中1出现的次数
class Solution:
    def NumberOf1Between1AndN_Solution(self, n):
        count = 0
        i = 1
        while (n // i) != 0:
            current = (n // i) %10 # 当前位数字
            before = n // (i * 10) # 高位数字
            after = n - (n // i) * i # 低位数字
            if current == 0:
                # 如果为0，出现1的次数有高位决定，等于高位数字*当前位数
                count += before * i
            elif current == 1:
                # 如果为1，出现1的次数由高位和低位决定，高位*当前位+低位+1
                count += before * i + after + 1
            else:
                # 如果大于1，出现1的次数由高位决定，（高位数字+1）*当前位数
                count += (before + 1) * i
            i = i * 10
        return count
"""

"""
# 把数组排成最小的数
class Solution:
    def PrintMinNumber(self, numbers):
        if not numbers:
            return ''
        num = map(str, numbers)
        num.sort(lambda x, y: cmp(x + y, y + x))
        return ''.join(num)
"""

"""
# 丑数
class Solution:
    def GetUglyNumber_Solution(self, index):
        if index <= 0:
            return 0
        uglyList = [1]
        index2, index3, index5 = 0, 0, 0
        for i in range(index - 1):
            newUgly = min(uglyList[index2] * 2, uglyList[index3] * 3, uglyList[index5] * 5)
            uglyList.append(newUgly)
            if newUgly % 2 == 0:
                index2 += 1
            if newUgly % 3 == 0:
                index3 += 1
            if newUgly % 5 == 0:
                index5 += 1
        return uglyList[-1]
"""

"""
# 第一个出现一次的字符
class Solution:
    def FirstNotRepeatingChar(self, s):
        if not s:
            return -1
        s_num = {x: 0 for x in set(s)}
        print(s_num)
        for i in s:
            s_num[i] += 1
        for i, j in enumerate(s):
            if s_num[j] == 1:
                return i
s = Solution()
print(s.FirstNotRepeatingChar('google'))
"""

"""
# 数组中的逆序对
count = 0
class Solution:
    def InversePairs(self, data):
        global count
        def MergeSort(lists):
            global count
            if len(lists) <= 1:
                return lists
            num = int(len(lists) / 2)
            left = MergeSort(lists[:num])
            right = MergeSort(lists[num:])
            print('left:', left)
            print('right:', right)
            r, l=0, 0
            result=[]
            while l<len(left) and r<len(right):
                if left[l] < right[r]:
                    result.append(left[l])
                    l += 1
                else:
                    result.append(right[r])
                    r += 1
                    count += len(left)-l
                    print('count:', count)
            print(result)
            result += right[r:]
            print(result)
            result += left[l:]
            print(result)
            return result
        MergeSort(data)
        return count % 1000000007
s = Solution()
print(s.InversePairs([1,2,3,4,5,6,7,0]))
"""

"""
# 两个链表的第一个公共结点
class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None
class Solution:
    def FindFirstCommonNode(self, pHead1, pHead2):
        # write code here
        if not pHead1 or not pHead2:
            return None
        count1, count2 = 0, 0
        node1, node2 = pHead1, pHead2
        while node1:
            count1 += 1
            node1 = node1.next
        while node2:
            count2 += 1
            node2 = node2.next
        node1, node2 = pHead1, pHead2
        if count1 >= count2:
            l = count1 - count2
            num = count2
            while l:
                node1 = node1.next
                l -= 1
        else:
            l = count2 - count1
            num = count1
            while l:
                node2 = node2.next
                l -= 1
        while node1 != None and node2 != None and node1 != node2:
            node1 = node1.next
            node2 = node2.next
        return node1
"""

"""
# 数字在排序数组中出现的次数
# 也可用二分查找变形做
class Solution:
    def GetNumberOfK(self, data, k):
        if k not in data:
            return 0
        i = [x for x in data if x == k]
        return len(i)
s = Solution()
print(s.GetNumberOfK([1, 2, 3, 3, 3, 3, 4, 5], 3))
"""

"""
# 二叉树的深度
class Solution:
    def TreeDepth(self, pRoot):
        if pRoot == None:
            return 0
        return max(self.TreeDepth(pRoot.left), self.TreeDepth(pRoot.right)) + 1
"""

"""
# 平衡二叉树
class Solution:
    def IsBalanced_Solution(self, pRoot):
        # 每个结点遍历不止一次
        if not pRoot:
            return True
        left = self.treedepth(pRoot.left)
        right = self.treedepth(pRoot.right)
        if abs(left - right) > 1:
            return False
        return (self.IsBalanced_Solution(pRoot.left) and self.IsBalanced_Solution(pRoot.right))

    def treedepth(self, pRoot):
        if not pRoot:
            return 0
        return max(self.treedepth(pRoot.left), self.treedepth(pRoot.right)) + 1
    # 想法：后序遍历时记录结点深度，作比较即可
"""

"""
# 数组中只出现一次的数字
# 异或^，将数组按某一位为1或是为0分开，再异或
class Solution:
    # 返回[a,b] 其中ab是出现一次的两个数字
    def FindNumsAppearOnce(self, array):
        if len(array) == 0 or array == 1:
            return array
        a = [x for x in array if array.count(x) == 1]
        return a
"""

"""
# 和为S的连续整数序列
class Solution:
    def FindContinuousSequence(self, tsum):
        num = tsum
        result = []
        for i in range(1, num + 1):
            sum = 0
            for j in range(i, num + 1):
                sum += j
                if sum == tsum:
                    l = [x for x in range(i, j + 1)]
                    if len(l) >1:
                        result.append(l)
                    break
                if sum > tsum:
                    break
        return result
s = Solution()
print(s.FindContinuousSequence(3))
"""

"""
# 和为s的两个数
class Solution:
    def FindNumbersWithSum(self, array, tsum):
        small, big = 0, len(array) - 1
        s = []
        while small < big:
            sum_a = array[small] + array[big]
            if sum_a == tsum:
                s.append([array[small], array[big]])
            if sum_a > tsum:
                big -= 1
            else:
                small += 1
        if len(s) > 0:
            mul = [x[0] * x[1] for x in s]
            arg = mul.index(min(mul))
            s = s[arg]
        return s
s = Solution()
print(s.FindNumbersWithSum([1,2,4,5,10,11,15], 15))
"""

"""
# 左旋转字符串
# 原理： YX = (XTYT)T，字符串转置
class Solution:
    def LeftRotateString(self, s, n):
        if len(s) == 0 or n == 0:
            return s
        if n > len(s):
            n = n % len(s)
        x1 = [s[i] for i in range(n)]
        x2 = [s[i] for i in range(n, len(s))]
        x2.extend(x1)
        return ''.join(x2)
s = Solution()
print(s.LeftRotateString('abcXYZdef', 3))
"""

"""
# 反转单词顺序列
class Solution:
    def ReverseSentence(self, s):
        if len(s) == 0 or s.isspace():
            return s
        sen = s.split()
        sentence = [sen[i] for i in range(-1, -(len(sen) + 1), -1)]
        return ' '.join(sentence)
s = Solution()
print(s.ReverseSentence('I am a student'))
print(s.ReverseSentence((' ')))
"""

"""
# 扑克牌顺子
class Solution:
    def IsContinuous(self, numbers):
        if len(numbers) > 5:
            return False
        Flag = False
        numbers = [x for x in numbers if x != 0]
        c = [numbers.count(x) for x in numbers]
        if sum(c) != len(c):
            return Flag
        if len(numbers) == 1:
            Flag = True
            return Flag
        numbers.sort()
        count = 0
        for i in range(len(numbers) - 1):
            x = numbers[i + 1] - numbers[i]
            if x != 1:
                count += (x - 1)
        if count == 5 - len(numbers):
            Flag = True
        return Flag
s = Solution()
print(s.IsContinuous([0,1,3,0,5]))
print(s.IsContinuous([1,0,0,0,6]))
"""

"""
# 孩子们的游戏（圆圈中最后剩下的数）
class Solution:
    def LastRemaining_Solution(self, n, m):
        if m <= 1 or n == 0:
            return -1
        kids = [x for x in range(n)]
        index = 0
        while len(kids) > 1:
            index += m - 1
            index = index % len(kids)
            kids.pop(index)
        return kids[0]
"""

"""
# 求1+2+3+...+n，不使用关键字及条件语句
class Solution:
    def __init__(self):
        self.sum = 0
    def Sum_Solution(self, n):
        # write code here
        self.sum += n
        s = (n > 0) and self.Sum_Solution(n - 1)
        return self.sum
s = Solution()
print(s.Sum_Solution(5))
"""

"""
# 不用加减乘除做加法
class Solution:
    def Add(self, num1, num2):
        a = (num1 ^ num2) & 0xFFFFFFFF
        b = (num1 & num2) << 1
        while b != 0:
            x, y = a, b
            a = (x ^ y) & 0xFFFFFFFF
            b = (x & y) << 1
        return a if a >> 31 == 0 else a - pow(2, 32)
s = Solution()
print(s.Add(5,17))
"""

"""
# 把字符串转换成整数
class Solution:
    def StrToInt(self, s):
        try:
            return int(s)
        except Exception as e:
            return 0
"""

"""
# 数组中重复的数字
class Solution:
    # 这里要特别注意~找到任意重复的一个值并赋值到duplication[0]
    # 函数返回True/False
    def duplicate(self, numbers, duplication):
        a = [numbers.count(x) for x in numbers]
        d = [numbers[i] for i, j in enumerate(a) if j > 1]
        print(d)
        if len(d) > 0:
            duplication[0] = d[0]
        if len(duplication):
            print(duplication[0])
            return True
        else:
            return False
s = Solution()
print(s.duplicate([2,1,3,0,4],[]))
"""

"""
# 构建乘积数组
class Solution:
    def multiply(self, A):
        B = [1, A[0]] # 下三角连乘
        C = [1, A[-1]] # 上三角连乘
        for i in range(1, len(A) - 1):
            B.append(B[i] * A[i])
            C.append((C[i] * A[len(A) - i - 1]))
        D = [B[i] * C[len(B) - i - 1] for i in range(len(B))]
        return D
s = Solution()
print(s.multiply([2, 3, 4]))
"""

"""
# 表示数值的字符串
class Solution:
    # s字符串
    def isNumeric(self, s):
        import re
        m_integer = re.compile(r'^[+|-]?\d+$')
        m_in_k = re.compile(r'^[+|-]?\d*[e|E][+|-]?\d+$')
        m_float = re.compile(r'^[+|-]?\d*\.\d+$')
        m_f_ke = re.compile(r'^[+|-]?\d*\.\d+[e|E][+|-]?\d+$')
        print(m_integer.match(s))
        print(m_float.match(s))
        print(m_in_k.match(s))
        print(m_f_ke.match(s))
        if m_integer.match(s) or m_float.match(s) or m_in_k.match(s) or m_f_ke.match(s):
            return True
        return False
s = Solution()
print(s.isNumeric('1+100'))
"""

"""
# 字符流中第一个不重复的字符
class Solution:
    # 返回对应char
    def __init__(self):
        self.s = ''
        self.dict = {}
    def FirstAppearingOnce(self):
        for i in self.s:
            if self.dict[i] == 1:
                return i
        return '#'
    def Insert(self, char):
        self.s += char
        if char in self.dict:
            self.dict[char] += 1
        else:
            self.dict[char] = 1
"""

"""
# 链表中换的入口结点
class Solution:
    def EntryNodeOfLoop(self, pHead):
        if not pHead:
            return pHead
        if not pHead.next:
            return pHead.next
        next_list = []
        while pHead:
            next_list.append(pHead)
            pHead = pHead.next
            if pHead in next_list:
                return pHead
"""

"""
# 删除链表中重复的点"""
class Solution:
    def deleteDuplication(self, pHead):
        if not pHead or not pHead.next:
            return pHead
        node = pHead
        while node:
            n_next = node.next
            if n_next.val == node:
                n = n_next.next
                while n.val == n_next.val:
                    n = n.next
                
"""
# 正则表达式
class Solution:
    # s, pattern都是字符串
    def match(self, s, pattern):
        if len(s) == 0 and len(pattern) == 0:
            return True
        if len(pattern) == 0 and len(s) != 0:
            return False
        if len(pattern) > 1 and pattern[1] == '*':
            if len(s) > 0 and (s[0] == pattern[0] or pattern[0] == '.'):
                # 匹配2个字符，匹配1个字符，匹配0个字符
                return (self.match(s[1:], pattern) or self.match(s[1:], pattern[2:]) or self.match(s, pattern[2:]))
            else:
                return self.match(s, pattern[2:])
        if len(s) != 0 and (pattern[0] == '.' or pattern[0] == s[0]):
            return self.match(s[1:], pattern[1:])
        return False
s = Solution()
print(s.match('bbbba', '.*a*a'))
"""

"""
# 二叉树的下一个节点
class TreeLinkNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None
        self.next = None
class Solution:
    def GetNext(self, pNode):
        if pNode == None:
            return pNode
        if pNode.right != None:
            pNode = pNode.right
            while pNode.left != None:
                pNode = pNode.left
            return pNode
        while pNode.next != None:
            if pNode == pNode.next.left:
                return pNode.next
            pNode = pNode.next
"""

"""
# 平衡二叉树
class Solution:
    def IsBalanced_Solution(self, pRoot):"""

"""
# m种颜色n个扇形
# n个扇形时，相邻扇形之间不能同色，共有m*(m - 1)^(n - 1)种染色方法，但由于An和A1相邻，应排除其同色的情况，即相当于两块合成一块扇形，n-1个扇形
# 的染色方法， 则an = m*(m - 1)^(n - 1)-an-1
def tuse(m, n):
    if m < 3 or n < 1:
        return False
    result = 0
    if n == 1:
        result = m
    elif n == 2:
        result = m * (m - 1)
    else:
        result = m * pow(m - 1, n - 1) - tuse(m, n - 1)
    return result"""

"""
# 找零问题
def coin(coins, price):
    coinUsed = {i: 0 for i in range(price + 1)}
    for cents in range(1, price + 1):
        minCoins = cents
        for value in coins:
            if value <= cents:
                temp = coinUsed[cents - value] + 1
                if temp < minCoins:
                    minCoins = temp
        coinUsed[cents] = minCoins
        print('面值{0}的硬币数目为{1}'.format(cents, coinUsed[cents]))
coin([25, 21, 10, 5, 1], 63)"""

"""
# 树的先中后序遍历，深度优先遍历，广度优先遍历
class tree:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None
class tree_op:
    def preOrderTraversal(self, root):
        l = []
        if root != None:
            l.extend(root.val)
            l.extend(self.preOrderTraversal(root.left))
            l.extend(self.preOrderTraversal(root.right))
        return l
    def midOrderTraversal(self, root):
        l = []
        if root != None:
            l.extend(self.midOrderTraversal(root.left))
            l.extend(root.val)
            l.extend(self.midOrderTraversal(root.right))
        return l
    def laterOrderTraversal(self, root):
        l = []
        if root != None:
            l.extend(self.laterOrderTraversal(root.left))
            l.extend(self.laterOrderTraversal(root.right))
            l.extend(root.val)
        return l
    def depthSearch(self, root):
        stack = []
        l = []
        stack.append(root)
        while len(stack) != 0:
            node = stack.pop()
            l.append(node.val)
            if node.right != None:
                stack.append(node.right)
            if node.left != None:
                stack.append(node.left)
        return l
    def breadthSearch(self, root):
        queue = []
        l = []
        queue.append(root)
        while len(queue) != 0:
            node = queue.pop(0)
            l.append(node.val)
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        return l

a = tree('a')
b = tree('b')
c = tree('c')
d = tree('d')
e = tree('e')
f = tree('f')
g = tree('g')
a.left = b
a.right = c
b.left = d
b.right = e
c.left = f
c.right = g
s = tree_op()
print(s.preOrderTraversal(a))
print(s.depthSearch(a))
print(s.breadthSearch(a))
print(s.midOrderTraversal(a))"""

"""
# 排序
def mergeSort(data):
    if len(data) <= 1:
        return data
    lenth = len(data) // 2
    left = mergeSort(data[: lenth])
    right = mergeSort(data[lenth :])
    result = []
    l, r = 0, 0
    while len(left) > l and len(right) > r:
        if left[l] < right[r]:
            result.append(left[l])
            l += 1
        else:
            result.append(right[r])
            r += 1
    result += right[r :]
    result += left[l :]
    return result
def quickSort(data):
    if len(data) <= 1:
        return data
    mid = data[0]
    left = quickSort([x for x in data if x < mid])
    right = quickSort([x for x in data if x > mid])
    return left + [mid] + right
data = [2,3,1,6]
print(mergeSort(data))
print(quickSort(data))
"""