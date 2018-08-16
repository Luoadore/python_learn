# coding: utf-8
# sort

# 插入排序，O(n^2)，稳定
# 基本操作：将一个数据插入到已经排好序的有序数据中
def insert_sort(lists):
	count = len(lists)
	for i in range(1, count):
		key = lists[i]
		j = i - 1
		while j >= 0:
			if lists[j] > key:
				lists[j + 1] = lists[j]
				lists[j] = key
			j -= 1
	return lists

# 希尔排序，O(nlogn)，不稳定
# 基本操作：按下标的增量分组，最每组使用直接插入排序
def shell_sort(lists):
	count = len(lists)
	step = 2
	group = count // step
	while group > 0:
		for i in range(0, group):
			j = i + group
			while j < count:
				k = j - group
				key = lists[j]
				while k >= 0:
					if lists[k] > key:
						lists[k + group] = lists[k]
						lists[k] = key
					k -= group
				j += group
		group /= step
	return lists

# 冒泡排序，O(n^2)，稳定
# 重复比较两个元素，顺序错误就互换
def bubble_sort(lists):
	count = len(lists)
	for i in range(count):
		for j in range(i + 1, count):
			if lists[i] > lists[j]:
				lists[i], lists[j] = lists[j], lists[i]
	return lists

# 快速排序，O(nlogn)，不稳定
# 递归进行，一趟将数据分为两部分，一部分比另一部分的所有数都小
def quick_sort(lists, left, right):
	if left >= right:
		return lists
	key = lists[left]
	low = left
	high = right
	while left < right:
		while left < right and lists[right] >= key:
			right -= 1
		lists[left] = lists[right]
		while left < right and lists[left] <= key:
			left += 1
		lists[right] = lists[left]
	lists[right] = key
	# print(lists)
	quick_sort(lists, low, left - 1)
	quick_sort(lists, right + 1, high)
	return lists
# print(quick_sort([9,8,7,4,3,6], 0, 5))

def quick_sort(lists):
	if len(lists) <= 1:
		return lists
	mid = lists.pop(0)
	left = quick_sort([x for x in lists if x <= mid])
	right = quick_sort([x for x in lists if x > mid])
	return left + [mid] + right

# 直接选择排序
# 基本思想：在待排序记录的r[i]~r[n]中选最小的记录，将其与r[i]交换
def select_sort(lists):
	count = len(lists)
	for i in range(count):
		min_value = i
		for j in range(i + 1, count):
			if lists[min_value] > lists[j]:
				min_value = j
		lists[min_value], lists[i] = lists[i], lists[min_value]
	return lists

# 堆排序，O(nlogn)，不稳定
# 大根堆，完全二叉树，每个节点的值都不大于父节点的值
# 堆中的结点按层进行编号，映射到数组
# arr[i] >= arr[2i+1] && arr[i] >= arr[2i+2]
# 基本思想：构造大顶堆，将堆顶元素与末尾元素交换，重建堆，从下到上，从右到左
def adjust_heap(lists, i, size):
	# 结构调整，使父节点的值大于子节点
	lchild = 2 * i + 1
	rchild = 2 * i + 2
	max = i
	if i < size // 2:
		if lchild < size and lists[lchild] > lists[max]:
			max = lchild
		if rchild < size and lists[rchild] > lists[max]:
			max = rchild
		if max != i:
			lists[max], lists[i] = lists[i], lists[max]
			adjust_heap(lists, max, size)

def build_heap(lists, size):
	# 构造堆，O(n)
    for i in range(0, (size // 2))[::-1]:
	    adjust_heap(lists, i, size)

def heap_sort(lists):
	size = len(lists)
	build_heap(lists, size)
	for i in range(0, size)[::-1]:
		lists[0], lists[i] = lists[i], lists[0]
		adjust_heap(lists, 0, i)

# 归并排序，O(nlogn)，稳定
# 分治，空间复杂度O(n)，两个指针
def merge_sort(lists):
	if len(lists) <=1:
		return lists
	lenth = len(lists) // 2
	left = merge_sort(lists[: lenth])
	right = merge_sort(lists[lenth :])
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