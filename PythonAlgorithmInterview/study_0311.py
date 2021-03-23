import collections
import functools
import re
import sys
from typing import List


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


class Solution:

    # leetcode.com/problems/add-two-numbers
    def addTwoNumbers(self, l1: ListNode, l2: ListNode) -> ListNode:
        root = head = ListNode(0)

        carry = 0
        while l1 or l2 or carry:
            sum = 0
            # 두 입력값의 합 계산
            if l1:
                sum += l1.val
                l1 = l1.next
            if l2:
                sum += l2.val
                l2 = l2.next

            # 몫과 나머지 계산
            carry, val = divmod(sum + carry, 10)
            head.next = ListNode(val)
            head = head.next
        return root.next

    # leetcode.com/problems/swap-nodes-in-pairs
    def swapPairs(self, head: ListNode) -> ListNode:
        root = prev = ListNode(None)
        prev.next = head
        while head and head.next:
            # b가 a를 가리키도록 할당
            b = head.next
            head.next = b.next
            b.next = head

            prev.next = b

            head = head.next
            prev = prev.next.next

        return root.next


if __name__ == '__main__':
    s = Solution()
    # print(s.addTwoNumbers(ListNode([2,4,3]),ListNode([5,6,4])))
    print(s.swapPairs(ListNode[1,2,3,4]))