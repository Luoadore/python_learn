# coding: utf-8

"""exercise 1
print('1024 * 768 =', 1024 * 768)"""

"""exercise 2
print('n =', 123)
print('f =', 456.789)
print('s1 =', 'Hello, world')
print('s2 =', 'Hello, \'Adam\'')
print('s3 =', r'Hello, "Bart"')
print('s4 =', r'''Hello,
 Lisa!''')"""

"""exercise 3
a = 72
b = 85
imp = (b - a) / a * 100
print('Score has imporved by %.1f %%' % imp)"""

"""exercise 4
L = [
    ['Apple', 'Google', 'Mircrosoft'],
    ['Java', 'Python', 'Rudy', 'PHP'],
    ['Adam', 'Bart', 'Lisa']
]
print(L[0][0])
print(L[1][1])
print(L[2][2])"""

"""exercise 5
h = 1.75
w = 80.5
bmi = w / (h * h)
if bmi < 18.5:
 print('过轻')
elif bmi < 25:
 print('正常')
elif bmi < 28:
 print('过重')
elif bmi < 32:
 print('肥胖')
else:
 print('严重肥胖')"""

"""exercise 6
L = ['Bart', 'Lisa', 'Adam']
for name in L:
 print('Hello, %s!' % name)"""

"""exercise 7
n1 = 255
n2 = 1000
print(hex(n1))
print(hex(n2))"""

"""exercise 8
import math
def quadratic(a, b, c):
 d = b * b - 4 * a * c
 print(2 * a)
 if d < 0:
  print('No solution.')
 sol_1 = (- b + math.sqrt(d)) / (2 * a)
 sol_2 = (- b - math.sqrt(d)) / (2 * a)
 return sol_1, sol_2

print(quadratic(2, 3, 1))
print(quadratic(1, 3, -4))"""

"""exercise 9
# hanoi, need more exercise!!!!!!!!!!!!!!!!!!!!!!!
def move(n, a, b, c):
 if n == 1:
  print('move', a, '-->', c)
 else:
  move(n-1, a, c, b)
  move(1, a, b, c)
  move(n-1, b, a, c)

move(3, 'A', 'B', 'C')"""

"""exercise 10
L1 = ['Hello', 'World', 18, 'Apple', None]
L2 = [x.lower() for x in L1 if isinstance(x, str)]
print(L2)"""

"""exercise 11
# 杨辉三角
def triangles():
 l1 = [1]
 while(True):   #此行必有
     yield l1
     l1 = [1] + [x + y for x, y in zip(l1[:-1], l1[1:])] + [1]

n = 0
for t in triangles():
 print(t)
 n = n + 1
 if n == 10:
  break"""

"""exercise 12
def normalize(name):
 return name.capitalize()

L1 = ['adam', 'LISA', 'barT']
L2 = list(map(normalize, L1))
print(L2)
from functools import reduce
def prod(l):
 r = reduce(lambda x, y: x * y, l)
 return r
print('3 * 5 * 7 * 9 =', prod([3, 5, 7, 9]))
from functools import reduce
def f(x, y):
 if y >= 1:
  r = x * 10 + y
 else:
  r = x + y
 return r
def str2num(s):
 return {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '.': 0.1}[s]
def isnum(s):
 s1 = list(map(str2num, s))
 print(s1)
 global dou
 dou = 1
 if 0.1 in s1:
  i = s1.index(0.1)
  l = len(s1)
  for j in range(1, l - i):
   dou = dou * 0.1
   print(dou)
   s1[i + j] = s1[i + j] * dou
  s1.pop(i)
  print(s1)
 return s1
print(reduce(f, isnum('123.456')))"""

"""exercise 13
# 回数
def is_palindrome(n):
 return str(n) == str(n)[::-1] # [::-1] 翻转
def is_palindrome(n):
 m, i = 0, n
 while i:
  m = m * 10 + i % 10
  i = i // 10
 return n == m
output = filter(is_palindrome, range(1, 10000))
print(list(output))"""

""" exercise 14
L = [('Bob', 75), ('Adam', 92), ('Bart', 66), ('Lisa', 88)]
def by_name(t):
    return t[0]
L2 = sorted(L, key = by_name)
print(L2)
def by_score(t):
    return t[1]
L3 = sorted(L, key = by_score, reverse = True)
print(L3)"""

""" exercise 15
# decorator
import functools
def log(text):
    def decorator(func):
        @functools.wraps(func)
        def wrap(*args, **kw):
            if text != None:
                print(text)
            print('call %s' % func.__name__)
            return func(*args, **kw)
        return wrap
    return decorator

@log('execute')
def now():
    print('2018-03-04')

now()
# 接受参数有无字符串的情况：isinstance(text, types.FunctionType)"""

""" exercise 16
# partial
import functools
int2 = functools.partial(int, base = 2)
print(int2('10010'))
print(int('10010'))"""

"""exercise 17
class Screen:
    @property
    def width(self):
        return self._width
    @width.setter
    def width(self, val):
        self._width = val

    @property
    def height(self):
        return self._height
    @height.setter
    def height(self, val):
        self._height = val

    @property
    def resolution(self):
        return self._width * self._height

s = Screen()
s.width = 1024
s.height = 768
print(s.resolution)
assert s.resolution == 786432, '1024 * 786 = %d ?' % s.resolution"""

"""exercise 18
# 佩波那契数列
class Fib:
    def __init__(self):
        self.a, self.b = 0, 1
    def __iter__(self):
        return self # 实例本身就是迭代对象，故返回自己
    def __next__(self):
        self.a, self.b = self.b, self.a + self.b
        if self.a > 100000:
            raise StopIteration();
        return self.a

for n in Fib():
    print(n)"""

""" exercise 19"""