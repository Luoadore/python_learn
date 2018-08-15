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

# 希尔排序，不稳定
# 基本操作

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
    print(lists)
    quick_sort(lists, low, left - 1)
    quick_sort(lists, right + 1, high)
    return lists
#print(quick_sort([9,8,7,4,3,6], 0, 5))

def quickSort(data):
    if len(data) <= 1:
        return data
    mid = data[0]
    left = quickSort([x for x in data if x < mid])
    right = quickSort([x for x in data if x > mid])
    return left + [mid] + right
print(quickSort([9,8,7,4,3,6]))