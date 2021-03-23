import collections
import functools
import re
import sys
from typing import List


class Solution:
    # leetcode.com/problems/valid-parentheses/
    def isValid(self, s: str) -> bool:
        stack = []
        table = {
            ')': '(',
            '}': '{',
            ']': '['
        }

        for char in s:
            if char not in table:
                stack.append(char)
            elif not stack or table[char] != stack.pop():
                return False

        return len(stack) == 0


if __name__ == '__main__':
    s = Solution()
    print(s.isValid('(){}[]'))