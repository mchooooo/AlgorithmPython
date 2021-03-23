import sys


def isPrimeNumber(n):
    end = (int)(n**0.5)
    for i in range(2, end+1):
        if (n % i == 0):
            return False
    return n

m, n = map(int, sys.stdin.readline().split())

for i in range(m, n+1):
    if i == 1:
        continue
    k = isPrimeNumber(i)
    if k:
        print(k)