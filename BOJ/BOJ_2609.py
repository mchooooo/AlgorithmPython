import sys

def gcd(n, m):

    while 1:
        r = n%m
        if r == 0:
            return m

        n = m
        m = r


big, small = map(int, sys.stdin.readline().split())

if small > big:
    big, small = small, big

a = gcd(big,small)
b = (big * small) / a
print(a)
print(int(b))