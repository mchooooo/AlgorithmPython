import collections
import functools
import re
import sys
import heapq
from typing import List


class Solution:

    # leetcode.com/problems/remove-duplicate-letters
    def removeDuplicateLetters(self, s: str) -> str:
        # 재귀 방법
        # for char in sorted(set(s)):
        #     suffix = s[s.index(char):]
        #     # 전체 집합과 접미사 집합이 같으면 분리 진행
        #     if set(s) == set(suffix):
        #         return char + self.removeDuplicateLetters(suffix.replace(char,''))
        #
        # return ''

        # 스택 이용
        counter, seen, stack = collections.Counter(s), set(), []

        for char in s:

            counter[char] -= 1
            if char in seen:
                continue
            # 뒤에 붙일 문자가 남아 있다면 스택에서 제거
            while stack and char < stack[-1] and counter[stack[-1]] > 0:
                seen.remove(stack.pop())
            stack.append(char)
            seen.add(char)
        return ''.join(stack)

    # leetcode.com/problems/daily-temperatures
    def dailyTemperatures(self, T: List[int]) -> List[int]:
        # 스택을 이용한 풀이
        answer = [0] * len(T)
        stack = []

        for i, cur in enumerate(T):
            # 현재 온도가 스택 값보다 높다면 처리
            while stack and cur > T[stack[-1]]:
                last = stack.pop()
                answer[last] = i - last
            stack.append(i)
        return answer


    # leetcode.com/problems/jewels-and-stones
    def numJewelsInStones(self, jewels: str, stones: str) -> int:
        # 해쉬테이블 이용
        # freqs = {}
        # count = 0
        # for char in stones:
        #     if char not in freqs:
        #         freqs[char] = 1
        #     else:
        #         freqs[char] += 1
        #
        # for char in freqs:
        #     if char in jewels:
        #         count += freqs[char]
        #
        # return count

        # defaultdict 이용
        # freqs = collections.defaultdict(int)
        # count = 0
        #
        # for char in stones:
        #    freqs[char] += 1
        #
        # for char in jewels:
        #     count += freqs[char]
        # return count

        # Counter 이용
        # freqs = collections.Counter(stones)
        # count = 0
        #
        # for char in jewels:
        #     count += freqs[char]
        #
        # return count

        # 컴프리헨션
        return sum(s in jewels for s in stones)

    #leetcode.com/problems/longest-substring-without-repeating-characters
    def lengthOfLongestSubstring(self, s: str) -> int:
        used = {}
        max_length = start = 0
        for index, char in enumerate(s):
            # 이미 등장했던 문자라면 'start' 위치 갱신
            if char in used and start <= used[char]:
                start = used[char] + 1
            else: # 최대 부분 문자열 길이 갱신
                max_length = max(max_length, index - start + 1)

            # 현재 문자의 위치 삽입
            used[char] = index
        return max_length


    # leetcode.com/problems/top-k-frequent-elements
    def topKFrequent(self, nums: List[int], k: int) -> List[int]:
        freqs = collections.Counter(nums)
        freqs_heap = []

        # 힙에 음수로 삽입
        for f in freqs:
            heapq.heappush(freqs_heap, (-freqs[f], f)) # -3 : 1, -2 : 2, -1 : 3

        topk = []
        for _ in range(k):
            topk.append(heapq.heappop(freqs_heap)[1])
        return topk

if __name__ == '__main__':
    s = Solution()
    # print(s.removeDuplicateLetters('cbacdcbc'))
    # print(s.dailyTemperatures([73,74,75,71,69,72,76,73]))
    # print(s.numJewelsInStones('aA','aAAbbbbbbb'))
    # print(s.lengthOfLongestSubstring('abcabcbb'))
    print(s.topKFrequent([1,1,1,2,2,3], 2))