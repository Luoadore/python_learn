# coding: utf-8

# 头条 

# 去除字符串中的空格。O(n)复杂度 ##########################################################
# 两个指针
def select_array(array):
     string = [x for x in array]
     i, j = 0, 0
     l = len(array)
     while j < l:
          if string[j] != ' ':
               string[i] = string[j]
               i += 1
          j += 1
     return ''.join(string[: i]), i

print(select_array('i need a girl'))
print(select_array(' a b c'))

# 输出树的最小深度 ########################################################################
# 递归
# 根节点到达最近的叶子节点的路径长度
def minimum_depth(root):
     if root == None:
          return 0
     if root.left == None:
          return minimum_depth(root.right) + 1
     if root.right == None:
          return minimum_depth(root.left) + 1
     return min(minimum_depth(root.left), minimum_depth(root.right)) + 1

# 非递归
# 树的层次遍历，current、next代表当前层和下一层的节点数，
# current=0时标明此层遍历完成，高度加1
# 停止条件：某节点左右孩子均为空
def min_depth(root):
    if root == None:
        return 0
    queue = []
    queue.append(root)
    current, next_layer = 1, 0
    height = 0
    while len(queue) != 0:
        node = queue.pop(0)
        current -= 1
        if not node.left and not node.right:
            height += 1
            break
        if node.left:
            queue.append(node.left)
            next_layer += 1
        if node.right:
            queue.append(node.right)
            next_layer += 1
        if current == 0:
            height += 1
            current, next_layer = next_layer, 0
    return height

# top k ##################################################################################
# 堆排序，建立一个k个元素的堆
# 遍历数组维护这个堆
# 时间复杂度，O(nlogK)
## 最小的k个用最大堆，最大的k个用最小堆
def topk(lists, k):

    def adjust_heap(lists, i, size):
        lchild = 2 * i + 1
        rchild = 2 * i + 2
        max_i = i
        if i < size // 2:
            if lchild < size and lists[lchild] > lists[max_i]:
                max_i = lchild
            if rchild < size and lists[rchild] > lists[max_i]:
                max_i = rchile
            if max_i != i:
                lists[max_i], lists[i] = lists[i], lists[max_i]
                adjust_heap(lists, max_i, size)

    def build_heap(lists, size):
        for i in range(size // 2)[::-1]:
            adjust_heap(lists, i, size)

    def set_top(lists, top):
        lists[0] = top
        adjust_heap(lists, 0, len(lists))

    top = [lists[i] for i in range(k)]
    build_heap(top, len(top))
    for i in range(k, len(lists)):
        if lists[i] < top[0]:
            set_top(top, lists[i])
    return top
# print(topk([1, 17, 3, 4, 5, 6, 7, 16, 9, 10, 11, 12, 13, 14, 15, 8], 4))

# 快排
# 分治
# partition返回第一次快排的位置，之后再与k比较
def top_k(lists, k):
    def partition(lists, low, high):
    	if len(lists) != 0 and low < high:
    		flag = lists[low]
    		while low < high:
    			while low < high and lists[high] >= flag:
    				high -= 1
    			lists[low] = lists[high]
    			while low < high and lists[low] <= flag:
    				low += 1
    			lists[high] = lists[low]
    		lists[low] = flag
    		return low
    	return 0

    low, high = 0, len(lists) - 1
    index = partition(lists, low, high)
    print(index)
    while index != k - 1:
    	if index > k - 1:
    		high = index - 1
    		index = partition(lists, low, high)
    	if index < k - 1:
    		low = index + 1
    		index = partition(lists, low, high)
    	print(index)
    return lists[: k]

print(top_k([1, 17, 3, 4, 5, 6, 7, 16, 9, 10, 11, 12, 13, 14, 15, 8], 4))