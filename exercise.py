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

""" exercise 19
from enum import Enum
Month = Enum('Month', ('Jan', 'Feb', 'Mar', 'Apr', 'May'))
for name, member in Month.__members__.items():
    print(name, "=>", member, ',', member.value)"""

"""exercise 20
# logging
import logging
logging.basicConfig(level = logging.INFO)

s = '0'
n = int(s)
logging.info('n = %d' % n)
print(10 / n)"""

"""exercise 21
# 单元测试
import unittest

# 需要测试的代码
class Dict(dict):
    def __init__(self, **kw):
        super().__init__(**kw)
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(r"'Dict' has no attribute '%s'" % key)
    def __setattr__(self, key, value):
        self[key] = value
##########################

class TestDict(unittest.TestCase):
    def test_init(self):
        d = Dict(a = 1, b = 'test')
        self.assertEqual(d.a, 1)
        self.assertEqual(d.b, 'test')
        self.assertTrue(isinstance(d, dict))
    def test_key(self):
        d = Dict()
        d['key'] = 'value'
        self.assertEqual(d.key, 'value')
    def test_attr(self):
        d = Dict()
        with self.assertRaises(KeyError):
            value = d['empty']
    def test_attrerror(self):
        d = Dict()
        with self.assertRaises(AttributeError):
            value = d.empty
    def setUp(self):
        print('setUp...')
    def tearDown(self):
        print('tearDown...')
if __name__ == '__main__':
    unittest.main()"""

"""exercise 22
def fact(n):"""
"""
    This is a function calculating the n!.

    >>> fact(1)
    1
    >>> fact(5)
    120
    >>> fact(0)
    Traceback (most recent call last):
        ...
    ValueError: Value error.
    """
"""
    if n < 1:
        raise ValueError('Value error.')
    if n == 1:
        return 1
    return n * fact(n - 1)
if __name__ == '__main__':
    import doctest
    doctest.testmod()"""

"""exercise 23
import os
# s = [x for x in os.listdir('F:\codestore\python_learn')]
# print(s)
def Findfile(path, s):
    file = [x for x in os.listdir(path) if os.path.isfile(os.path.join(path, x))]
    folder = [x for x in os.listdir(path) if os.path.isdir(os.path.join(path, x))]
    print(file)
    print(folder)
    for i in range(len(file)):
        if s in file[i]:
            print(file[i])
    for i in range(len(folder)):
        f_file = [x for x in os.listdir(os.path.join(path, folder[i]))]
        for j in range(len(f_file)):
            if s in f_file[j]:
                print(os.path.join(folder[i], f_file[j]))

mypath = 'F:\codestore\hyperspectral_exp_orz\\tf_try\GANs'
Findfile(mypath, 'cgan')"""

"""exercise 24
import json

class Student:
 def __init__(self, name, age, score):
  self.name = name
  self.age = age
  self.score = score

def student2dict(std):
 return{
        'name': std.name,
        'age': std.age,
        'score': std.score
 }
s = Student('Bob', 20, 88)
print(json.dumps(s, default = student2dict))"""

"""exercise 25
from multiprocessing import Process
import os

# 子进程要执行的程序
def run_proc(name):
    print('Run child process %s (%s)...' % (name, os.getpid()))

if __name__ == '__main__':
    print('Parent process %s.' % os.getpid())
    p = Process(target = run_proc, args = ('test',))
    print('Child process will start.')
    p.start()
    p.join()
    print('Child process end.')

# 进程间的通信
from multiprocessing import Process, Queue
import os, time, random

# 写数据进程
def write(q):
    print('Process to write: %s' % os.getpid())
    for value in ['A', 'B', 'C']:
        print('Put %s to queue...' % value)
        q.put(value)
        time.sleep(random.random())

# 读
def read(q):
    print('Process to read: %s' % os.getpid())
    while True:
        value = q.get(True)
        print('Get %s from queue.' % value)

if __name__ == '__main__':
    # 父进程创建Queue，并传给各个子进程
    q = Queue()
    pw = Process(target = write, args = (q,))
    pr = Process(target = read, args = (q,))
    # 启动子进程pw， 写入：
    pw.start()
    # 启动子进程pr，读取：
    pr.start()
    # 等待pw结束
    pw.join()
    pr.terminate()"""

"""exercise 26
# 一句话实现列表变字典
a = [1, 2, 3, 4, 5]
b = ['a', 'b', 'c', 'd', 'e']
mydict = {x: y for x, y in zip(a, b)}
print(mydict)
print(type(mydict))"""

"""exercise 27
# 检查回文数
list_to_check = [1, 123, 121, 232, 234, 566, 123321]
def isHui(num):
    num_l = str(num)
    l = len(num_l)
    flag = 1
    if l == 1:
        return num
    else:
        mid = l // 2
        for i, j in zip(range(mid), range(-1, -(mid + 1), -1)):
            if num_l[i] != num_l[j]:
                flag = 0
        if flag == 1:
            return num
check = []
for num in list_to_check:
    if isHui(num) != None:
        check.append(isHui(num))
print(check)"""

"""exercise 28
# 正则表达式
import re
mail = re.compile(r'^(\w+)\@(\w+)\.com$')
a = mail.match('someone@gmail.com').groups()
print(a)"""

"""exercise 29
import re
from datetime import datetime, timezone, timedelta

def to_timestamp(dt_str, tz_str):
    dt = datetime.strptime(dt_str, '%Y-%m-%d %H:%M:%S')
    u = re.compile(r'UTC([+|-][0-9]+)\:00')
    d_utc = u.match(tz_str).group(1)
    dt_utc = dt.replace(tzinfo=timezone(timedelta(hours=int(d_utc))))
    return dt_utc.timestamp()
t1 = to_timestamp('2015-6-1 08:10:30', 'UTC+7:00')
assert t1 == 1433121030.0, t1
t2 = to_timestamp('2015-5-31 16:10:30', 'UTC-09:00')
assert t2 == 1433121030.0, t2

print('Pass')"""

"""exercise 30
import base64
def safe_base64_decode(s):
    if len(s) % 4 != 0:
        num = 4 - len(s) % 4
        for i in range(num):
            if type(s) == bytes:
                s += b'='
            else:
                s += '='
    return base64.b64decode(s)
assert b'abcd' == safe_base64_decode(b'YWJjZA=='), safe_base64_decode('YWJjZA==')
assert b'abcd' == safe_base64_decode(b'YWJjZA'), safe_base64_decode('YWJjZA')
print('pass')
"""

""" exercise 31
import struct
def bmpinfo(path):
    f = open(path, 'rb')
    s = f.read(30)
    s_unpack = struct.unpack('<ccIIIIIIHH', s)
    if s_unpack[0] == b'B' and (s_unpack[1] == b'A' or s_unpack[1] == b'M'):
        print('位图大小:', s_unpack[6], 'x', s_unpack[7])
        print('Number of colors:', s_unpack[-1])
    else:
        print('不是位图。')
bmpinfo('E:\star.bmp')
bmpinfo('E:\Git-2.12.0-64-bit.exe')
"""

""" exercise 32
import hashlib
def calc_md5(password):
    md5 = hashlib.md5()
    md5.update(password.encode('utf-8'))
    return md5.hexdigest()
def login(user, password):
    p = calc_md5(password + user + 'the-Salt')
    if p == db[user]:
        print('Register success!')
    else:
        print('User name or password wrong!')
def register(user, password):
    db[user] = calc_md5(password + user + 'the-Salt')
db = {}
register('bob', '123456')
register('alice', 'kristy')
register('mike', '1993girl')
print(db)
login('bob', '12345')
login('mike', '1993girl')
"""

""" exercise 33
# !!!!!!!NOT CORRECT YET!!!!!!
from html.parser import HTMLParser
import re

liststr = []
time_list = []
local_list = []

class MyHTMLParser(HTMLParser):
    tempstr = str()
    matchObg = False
    def handle_starttag(self, tag, attrs):
        m = re.compile(r'event\-\w')
        s = [x[1] for x in attrs if x[0] == 'class']
        if len(s) > 0:
            matchObj = m.match(s[0])
        print(matchObj)
        if matchObj:
            self.matchObg = True
            self.tempstr = ''
    def handle_endtag(self, tag):
        if self.matchObg:
            liststr.append(self.tempstr)
    def handle_data(self, data):
        if (data.isspace() == False):
            self.tempstr += data + '\t'

with open('E:/test.html', 'r') as f:
    html = f.readlines()
    par = MyHTMLParser()
    for i in range(500):
        par.feed(html[i])
    for value in liststr:
        print(value)"""

"""exercise 34
# coroutine 协程，一个线程执行
# generator
def consumer():
    r = ''
    while True:
        n = yield r
        if not n:
            return
        print('[CONSUMER] Consuming %s...' % n)
        r = '200 OK'

def produce(c):
    c.send(None)
    n = 0
    while n < 5:
        n = n + 1
        print('[PRODUCER] Producing %s...' % n)
        r = c.send(n)
        print('[PRODUCER] Consumer return: %s' % r)
    c.close()

c = consumer()
produce(c)
"""

""" exercise 35"""
# kafka
from kafka import KafkaProducer
from kafka import KafkaConsumer
import os
from sys import argv

producer = KafkaProducer(bootstrap_servers='127.0.0.1: 9092')
# Topic = test
producer.send('test', b'something')
producer.flush()
producer.close()

consumer = KafkaConsumer('test', bootstrap_servers='127.0.0.1: 9092')