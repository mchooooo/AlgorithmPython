import collections
import functools
import re
import sys
import heapq
from typing import List

# Definition for singly-linked list.
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class Solution:

    # leetcode.com/problems/sort-list/
    def mergeTwoLists(self, l1: ListNode, l2: ListNode) -> ListNode:
        if l1 and l2:
            if l1.val > l2.val:
                l1, l2 = l2, l1
            l1.next = self.mergeTwoLists(l1.next, l2)

        return l1 or l2


    def sortList(self, head: ListNode) -> ListNode:
        if not (head or head.next):
            return head

        # 런너 기법 사용
        half, slow, fast = None, head, head
        while fast and fast.next:
            half, slow, fast = slow, slow.next, fast.next.next
        half.next = None

        # 분할 재귀 호출
        l1 = self.sortList(head)
        l2 = self.sortList(slow)

        return self.mergeTwoLists(l1, l2)

    # leetcode.com/problems/largest-number/
    @staticmethod
    def to_swap(n1, n2) -> bool:
        return str(n1) + str(n2) < str(n2) + str(n1)


    def largestNumber(self, nums: List[int]) -> str:
        i = 1
        while i < len(nums):
            j = i

            while j > 0 and self.to_swap(nums[j-1], nums[j]):
                nums[j], nums[j-1] = nums[j-1], nums[j]
                j -= 1
            i += 1

        return str(int(''.join(map(str, nums))))

    # leetcode.com/problems/binary-search
    def search(self, nums: List[int], target: int) -> int:
        left, right = 0, len(nums)-1

        while left <= right:
            mid = (left + right) // 2

            if nums[mid] < target:
                left = mid+1
            elif nums[mid] > target:
                right = mid - 1
            else:
                return mid
        return -1

    # leetcode.com/problems/search-in-rotated-sorted-array/
    def search(self, nums: List[int], target: int) -> int:
        # 예외 처리
        if not nums:
            return -1

        # 최솟값을 찾아 피벗 설정
        left, right = 0, len(nums)-1
        while left < right:
            mid = left + (right - left) // 2
            if nums[mid] > nums[right]:
                left = mid + 1
            else:
                right = mid

        pivot = left

        # 피벗 기준 이진 검색
        left, right = 0, len(nums)-1
        while left <= right:
            mid = left + (right - left) // 2
            mid_pivot = (mid + pivot) % len(nums)

            if nums[mid_pivot] < target:
                left = mid + 1
            elif nums[mid_pivot] > target:
                right = mid - 1
            else:
                return mid_pivot
        return -1

    # leetcode.com/problems/sliding-window-maximum/
    def maxSlidingWindow(self, nums: List[int], k: int) -> List[int]:
        results = []
        window = collections.deque()
        current_max = float('-inf')
        for i, v in enumerate(nums):
            window.append(v)
            if i < k-1:
                continue
            # 새로 추가된 값이 기존 최댓값보다 큰 경우 교체
            if current_max == float('-inf'):
                current_max = max(window)
            elif v > current_max:
                current_max = v

            results.append(current_max)

            # 최댓값이 윈도우에서 빠지면 초기화
            if current_max == window.popleft():
                current_max = float('-inf')
        return results

    # leetcode.com/problems/minimum-window-substring/
    def minWindow(self, s: str, t: str) -> str:
        need = collections.Counter(t)
        # print(need)
        missing = len(t)
        left = start = end = 0

        # 오른쪽 포인터 이동
        for right, char in enumerate(s, 1):
            # print(right, char)
            missing -= need[char] > 0
            need[char] -= 1

            # 필요 문자가 0이면 왼쪽 포인터 이동 판단
            if missing == 0:
                while left < right and need[s[left]] < 0:
                    need[s[left]] += 1
                    left += 1
                if not end or right - left <= end - start:
                    start, end = left, right
                    need[s[left]] += 1
                    missing += 1
                    left += 1

            print(need, missing)
            print(left, right)
        return s[start:end]

    # leetcode.com/problems/longest-repeating-character-replacement/
    def characterReplacement(self, s: str, k: int) -> int:
        left = right = 0
        counts = collections.Counter()
        for right in range(1, len(s)+1):
            counts[s[right-1]] += 1
            # 가장 흔하게 등장하는 문자 탐색
            max_char_n = counts.most_common(1)[0][1]

            # k 초과시 왼쪽 포인터 이동
            if right - left - max_char_n > k:
                counts[s[left]] -= 1
                left += 1
            print(left, right)

        return right - left

    # leetcode.com/problems/best-time-to-buy-and-sell-stock-ii/
    def maxProfit(self, prices: List[int]) -> int:
        result = 0
        # 값이 오르는 경우 매번 그리디 계산
        for i in range(len(prices) - 1):
            if prices[i + 1] > prices[i]:
                result += prices[i + 1] - prices[i]
        return result

    # leetcode.com/problems/queue-reconstruction-by-height/
    def reconstructQueue(self, people: List[List[int]]) -> List[List[int]]:
        heap = []
        # 키 역순, 인덱스 삽입
        for person in people:
            heapq.heappush(heap, (-person[0], person[1]))

        # print(heap)

        result = []
        # 키 역순, 인덱스 추출
        while heap:
            person = heapq.heappop(heap)
            result.insert(person[1], [-person[0], person[1]])

        # print(result)
        return result

    # leetcode.com/problems/task-scheduler/
    def leastInterval(self, tasks: List[str], n: int) -> int:
        counter = collections.Counter(tasks)
        result = 0

        while True:
            sub_count = 0
            # 개수 순 추출
            for task, _ in counter.most_common(n+1):
                sub_count += 1
                result += 1

                counter.subtract(task)
                # 0이하인 아이테을 목록에서 완전히 제거
                counter += collections.Counter()

            if not counter:
                break
            result += n - sub_count + 1
        return result

    # leetcode.com/problems/gas-station/
    def canCompleteCircuit(self, gas: List[int], cost: List[int]) -> int:
        if sum(gas) < sum(cost):
            return -1

        start, fuel = 0, 0
        for i in range(len(gas)):
            # 출발점이 안 되는 지점 판별
            if gas[i] + fuel < cost[i]:
                start = i + 1
                fuel = 0
            else:
                fuel += gas[i] - cost[i]
        return start

    # leetcode.com/problems/assign-cookies/
    def findContentChildren(self, g: List[int], s: List[int]) -> int:
        g.sort()
        s.sort()

        child_i = cookie_j = 0
        # 만족하지 못 할 때까지 그리디 진행
        while child_i < len(g) and cookie_j < len(s):
            if s[cookie_j] >= g[child_i]:
                child_i+=1
            cookie_j += 1
        return child_i

    # leetcode.com/problems/majority-element/
    def majorityElement(self, nums: List[int]) -> int:
        if not nums:
            return None
        if len(nums) == 1:
            return nums[0]

        half = len(nums) // 2

        a = self.majorityElement(nums[:half])
        b = self.majorityElement(nums[half:])

        return [b, a][nums.count(a) > half]

    # leetcode.com/problems/different-ways-to-add-parentheses/
    def diffWaysToCompute(self, expression: str) -> List[int]:
        def compute(left, right, op):
            results = []
            for l in left:
                for r in right:
                    results.append(eval(str(l)+ op+ str(r)))
            return results
        if expression.isdigit():
            return [int[expression]]
        results = []

        for index, value in enumerate(expression):
            if value in "-*+":
                left = self.diffWaysToCompute(expression[:index])
                right = self.diffWaysToCompute(expression[index+1:])

                results.append(compute(left, right, value))
        return results

    # leetcode.com/problems/maximum-subarray/
    def maxSubArray(self, nums: List[int]) -> int:
        for i in range(1, len(nums)):
            if nums[i-1] > 0:
                nums[i] += nums[i - 1]

        return max(nums)

    # leetcode.com/problems/climbing-stairs/
    def climbStairs(self, n: int) -> int:
        dp = collections.defaultdict(int)
        dp[1] = 1
        dp[2] = 2
        for i in range(3, n+1):
            dp[i] = dp[i-1] + dp[i-2]
        return dp[n]

    # leetcode.com/problems/house-robber/
    def rob(self, nums: List[int]) -> int:
        if not nums:
            return 0
        if len(nums) <= 2:
            return max(nums)

        dp = collections.OrderedDict()
        dp[0], dp[1] = nums[0], max(nums[0], nums[1])
        for i in range(2, len(nums)):
            dp[i] = max(dp[i - 1], dp[i - 2] + nums[i])

        return dp.popitem()[1]


if __name__ == '__main__':
    s = Solution()
    # print(s.search([-1,0,3,4,9,12], 9))
    # print(s.maxSlidingWindow([1, 3, -1, -3, 5, 3 ,6, 7], 3))
    # print(s.minWindow('ADOBECODEBANC','ABC'))
    # print(s.characterReplacement('AAABBC',2))
    # print(s.reconstructQueue([[7, 0], [4, 4], [7, 1], [5, 0], [6, 1], [5, 2]]))
    print(s.majorityElement([2, 2, 1, 1, 1, 2, 2]))