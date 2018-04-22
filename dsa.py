# coding: utf-8

# 两个链表有相同的结点
class node():
    def __init__(self, x):
        self.val = x
        self.next = None
def findfirstcommonnode(anode, bnode):
    if not anode or not bnode:
        return None
    a_len, b_len = 0, 0
    nodea, nodeb = anode, bnode
    while nodea:
        a_len += 1
        nodea = nodea.next
    while nodeb:
        b_len += 1
        nodeb = nodeb.next
    print(a_len)
    print(b_len)
    if a_len >= b_len:
        l = a_len - b_len
        while l:
            anode = anode.next
            l -= 1
    else:
        l = b_len - a_len
        while l:
            bnode = bnode.next
            l -= 1
    while anode != bnode:
        anode = anode.next
        bnode = bnode.next
    return anode
a = node(1)
b = node(2)
c = node(3)
d = node(4)
e = node(5)
f = node(6)
g = node(7)
"""a.next = b
b.next = c
c.next = f
d.next = e
e.next = f
f.next = g
print(findfirstcommonnode(a, d).val)"""

# 链表中有环
def findRing(phead):
    slow = phead
    fast = phead
    while slow or fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow == fast:
            fast = phead
            break
    while slow != fast:
        slow = slow.next
        fast = fast.next
    return slow
a.next = b
b.next = c
c.next = d
d.next = e
e.next = f
f.next = g
g.next = c
print(findRing(a).val)

# 和为s的两个数 vs 和为s的序列
def find2sum(array, s):
    small, big = 0, len(array) - 1
    array = sorted(array)
    sum_all = []
    while small < big:
        m = array[small] + array[big]
        if m < s:
            small += 1
        elif m > s:
            big -= 1
        else:
            sum_all.append([array[small], array[big]])
            small += 1
    return sum_all
print(find2sum([1,2,4,7,10,11,15],15))

def findnsum(s):
    small, big = 1, 2
