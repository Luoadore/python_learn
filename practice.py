# coding:utf-8

def heap_adjust(lists, i, size):
	lchild = 2*i +1
	rchild = 2*i +2
	max_i = i
	if i < size // 2:
		if lchild < size and lists[lchild] > lists[max_i]:
			max_i = lchild
		if rchild < size and lists[rchild] > lists[max_i]:
			max_i = rchild
		if max_i != i:
			lists[i], lists[max_i] = lists[max_i], lists[i]
			heap_adjust(lists, max_i, size)

def build_heap(lists, size):
	for i in range(size // 2)[::-1]:
		heap_adjust(lists, i, size)

def heap_sort(lists):
	build_heap(lists, len(lists))
	for i in range(len(lists))[::-1]:
		lists[0], lists[i] = lists[i], lists[0]
		heap_adjust(lists, 0, i)
	return lists

print('ok')
print(heap_sort([2,4,1,6,2,8,7,9,0]))
print('done')

# 图的深度优先遍历（dfs）
# 应用：最大路径
def dfs(visit, index):
	pass

# 链表反转
class listnode:
	def __init__(self, x):
		self.val = x
		self.next = None
	def set_next(self, next):
		self.next = next
def reverse_list(phead):
	if phead == None or phead.next == None:
		return phead

	head = phead
	p_new = None
	while head != None:
		p_last = head.next
		head.next = p_new
		p_new = head
		head = p_last
	return p_new

# 判断单链表是否有环
# hash
def has_cycle(head):
	nodes = []
	node = head
	while node:
		if node in nodes:
			return True
		else:
			nodes.append(node)
			node = node.next
	return False

# fast slow
def has_cycle_2(head):
	slow = head
	fast = head
	while slow or fast.next:
		slow = slow.next
		fast = fast.next.next
		if slow == fast:
			fast = head
			break
	while slow != fast:
		slow = slow.next
		fast = fast.next
	return slow

# 归并排序
def mergesort(lists):
	if len(lists) <= 1:
		return lists
	mid = len(lists) // 2
	left = mergesort(lists[: mid])
	right = mergesort(lists[mid :])
	l, r = 0, 0
	result = []
	while l < len(left) and r < len(right):
		if left[l] < right[r]:
			result.append(left[l])
			l += 1
		else:
			result.append(right[r])
			r += 1
	result += right[r :]
	result += left[l :]
	return result

# 根据中序和前序重建二叉树
class treenode:
	def __init__(self, x):
		self.val = x
		self.left = None
		self.right = None
	def set_child(self, l, r):
		self.left = l
		self.right = r
# 递归
def reconstruct(pre, tin):
	if not pre or tin:
		return None
	root = treenode(pre.pop(0))
	index = tin.index(root.val)
	root.left = reconstruct(pre, tin[: index])
	root.right = reconstruct(pre, tin[index + 1 :])
	return root

# 平衡二叉树判断
def is_balance(root):
	if not root:
		return True
	left = treedepth(root.left)
	right = treedepth(root.right)
	if abs(left - right) > 1:
		return False
	return (is_balance(root.right)) and (is_balance(root.left))

def treedepth(root):
	if not root:
		return 0
	return max(treedepth(root.left), treedepth(root.right)) + 1

# 深度学习相关
def sigmoid_loss(softmax_re, labels):
	with tf.name_scope('loss'):
		log_tf = tf.log(softmax_re, name = 'log_name')
		entroy = tf.reduce_mean(labels * log_tf, reduction_indices=[1])
	return entroy

def acc(softmax_re, labels):
	with tf.name_scpoe('acc'):
		correct_prediction = tf.equal(tf.argmax(softmax_re, 1), tf.argmax(labels, 1))
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	return acc

def center_loss(features, labels, alpha, num_classes):
	len_features = features.get_shape()[1]

	centers = tf.get_variable('centers', [num_classes, len_features], dtype=tf.float32,
		initializer=tf.constant_initializer(0), trainable=False)
	labels = tf.reshape(labels, [-1])

	centers_batch = tf.gather(centers, labels)
	loss = tf.nn.l2_loss(features - centers_batch)

	diff = centers_batch - features

	unique_label, unique_idx, unique_count =  tf.unique_with_counts(labels)
	appear_times = tf.gather(unique_count, unique_idx)
	appear_times = tf.reshape(appear_times, [-1, 1])

	diff = diff / tf.cast((1 + appear_times), tf.float32)
	diff = alpha * diff

	centers_update_op = tf.scatter_sub(centers, labels, diff)

	return loss, centers, centers_update_op

def large_margin_loss(net, labels):
	loss

# 判断图是否联通
# 深度优先遍历
adjlists = [[1, 2, 3], [5, 6], [4], [2, 4], [1], [], [4]]
# 递归
def depth_first_search(adjlists, s):
	visited = []
	n = len(adjlists)
	for i in range(n):
		visited.append(False)
	dfs(adjlists, visited, s)
def dfs(adjlists, visited, v):
	visited[v] = True
	print(v, ' ', end='')
	for w in adjlists[v]:
		if not visited[w]:
			dfs(adjlists, visited, w)
depth_first_search(adjlists, 0)

# 非递归
def dfs_iterative(adjlists, s):
	stack = []
	stack.append(s)
	n = len(adjlists)
	visited = []
	for i in range(n):
		visited.append(False)
	while len(stack) > 0:
		v = stack.pop()
		if not visited[v]:
			visited[v] = True
			print(v, ' ', end='')

			stack_aux = []
			for w in adjlists[v]:
				if not visited[w]:
					stack_aux.append(w)
			while len(stack_aux) > 0:
				stack.append(stack_aux.pop())

