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

    # leetcode.com/problems/palindrome-linked-list
    def isPalindrome(self, head: ListNode) -> bool:
        # q = collections.deque()
        #
        # if not head:
        #     return True
        #
        # node = head
        # while node is not None:
        #     q.append(node.val)
        #     node = node.next
        #
        # while len(q) > 1:
        #     if q.popleft() != q.pop():
        #         return False
        #
        # return True


        # 런너 기법
        rev = None
        slow = fast = head
        # 런너를 이용해 역순 연결 리스트 구성
        while fast and fast.next:
            fast = fast.next.next
            rev, rev.next, slow = slow, rev, slow.next
        if fast:
            slow = slow.next

        # 팰린드롬 여부 확인
        while rev and rev.val == slow.val:

            slow, rev = slow.next, rev.next

        return not rev

    # leetcode.com/problems/merge-two-sorted-lists/
    def mergeTwoLists(self, l1: ListNode, l2: ListNode) -> ListNode:

        if (not l1) or (l2 and l1.val > l2.val):
            print(1)
            l1, l2 = l2, l1

        if l1:
            print(2)
            l1.next = self.mergeTwoLists(l1.next, l2)
        return l1

    # leetcode.com/problems/reverse-linked-list
    def reverseList(self, head: ListNode) -> ListNode:
        # 재귀
        def reverse(node: ListNode, prev: ListNode = None):
            if not node:
                return prev
            next, node.next = node.next, prev
            return reverse(next, node)
        # return reverse(head)

        # 반복구조
        node, prev = head, None

        while node:
            next, node.next = node.next, prev
            prev, node = node, next

        return prev


if __name__ == '__main__':
    s = Solution()
    # print(s.isPalindrome(ListNode([1, 2, 2, 1])))
    # print(s.mergeTwoLists(ListNode([1,2,4]),ListNode([1,3,4])))
    print(s.reverseList(ListNode([1,2,3,4,5])))