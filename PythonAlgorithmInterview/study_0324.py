import collections
import functools
import re
import sys
import heapq
from typing import List

class Solution:

    # leetcode.com/problems/number-of-islands
    def numIslands(self, grid: List[List[str]]) -> int:
        def dfs(i,j):
            # 더 이상 땅이 아닌 경우 종료
            if i < 0 or i >= len(grid) or j < 0 or j >= len(grid[0]) or grid[i][j] != '1':
                return

            grid[i][j] = 0
            # 동서남북 탐색
            dfs(i+1, j)
            dfs(i - 1, j)
            dfs(i, j + 1)
            dfs(i, j - 1)
            dq = collections.deque()

        count = 0
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if grid[i][j]=='1':
                    dfs(i, j)
                    count+=1

        return count

    # leetcode.com/problems/letter-combinations-of-a-phone-number
    def letterCombinations(self, digits: str) -> List[str]:
        def dfs(index, path):
            # 끝까지 탐색하면 백트래킹
            if len(path) == len(digits):
                result.append(path)
                return

            # 입력 값 자리 수 단위 반복
            for i in range(index, len(digits)):
                # 숫자에 해당하는 모든 문자열 반복
                for j in dic[digits[i]]:
                    # print(j)
                    dfs(i+1, path+j)

        # 에외 처리
        if not digits:
            return []

        dic = {"2": "abc", "3": "def", "4": "ghi", "5": "jkl", "6": "mno", "7": "pqrs", "8": "tuv", "9": "wxyz"}
        result = []
        dfs(0,"")

        return result


    # leetcode.com/problems/permutations
    def permute(self, nums: List[int]) -> List[List[int]]:
        result = []
        prev_element = []
        def dfs(elements):
            # 리프 노드일 때 결과 추가
            if len(elements) == 0:
                result.append(prev_element[:])

            # 순열 생성 재귀 호출
            for e in elements:
                next_elements = elements[:]
                next_elements.remove(e)

                prev_element.append(e)
                dfs(next_elements)
                prev_element.pop()

        dfs(nums)
        return result

    # leetcode.com/problems/combinations
    def combine(self, n: int, k: int) -> List[List[int]]:
        result = []
        def dfs(elements, start: int, k: int):
            if k == 0:
                result.append(elements[:])
                return
            # 자신 이전의 모든 값을 고정하여 재귀 호출
            for i in range(start, n+1):
                elements.append(i)
                dfs(elements, i + 1, k -1)
                elements.pop()

        dfs([], 1, k)
        return result


    # leetcode.com/problems/combination-sum/
    def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
        result = []
        def dfs(csum, index, path):
            # 종료 조건
            if csum < 0:
                return
            if csum == 0:
                result.append(path)
                return

            # 자신부터 하위 원소 까지의 나열 재귀 호출
            for i in range(index, len(candidates)):

                dfs(csum - candidates[i], i, path + [candidates[i]])

        dfs(target, 0, [])
        return result

    # leetcode.com/problems/subsets
    def subsets(self, nums: List[int]) -> List[List[int]]:
        results = []

        def dfs(index, path):
            # 매번 결과 추가
            results.append(path)

            # 경로를 만들면서 dfs
            for i in range(index, len(nums)):
                dfs(i+1, path + [nums[i]])

        dfs(0, [])
        return results

    # leetcode.com/problems/reconstruct-itinerary/
    def findItinerary(self, tickets: List[List[str]]) -> List[str]:
        # DFS 풀이
        graph = collections.defaultdict(list)
        # 그래프 순서대로 구성
        for a, b in sorted(tickets):
            graph[a].append(b)

        route = []
        def dfs(a):
            # 첫 번째 값을 읽어 어휘 순 방문
            while graph[a]:
                dfs(graph[a].pop(0))
            route.append(a)

        dfs('JFK')
        return route[::-1]

    # leetcode.com/problems/course-schedule
    def canFinish(self, numCourses: int, prerequisites: List[List[int]]) -> bool:
        graph = collections.defaultdict(list)
        # 그래프 구성
        for x, y in prerequisites:
            graph[x].append(y)

        traced = set()
        visited = set()

        def dfs(course):
            # 순환 구조이면 False
            if course in traced:
                return False
            # 이미 방문했던 노드이면 True
            if course in visited:
                return True

            traced.add(course)
            for y in graph[course]:
                if not dfs(y):
                    return False

            # 탐색 종료 후 순환 노드 삭제
            traced.remove(course)
            # 탐색 종료 후 방문 노드 추가
            visited.add(course)

            return True

        # 순환 구조 판별
        for x in list(graph):
            if not dfs(x):
                return False

        return True


if __name__ == '__main__':
    s = Solution()
    # print(s.numIslands([
  # ["1","1","1","1","0"],
  # ["1","1","0","1","0"],
  # ["1","1","0","0","0"],
  # ["0","0","0","0","0"]
# ]))
#     print(s.letterCombinations("23"))
#     print(s.permute([1, 2, 3]))
#     print(s.combine(4, 2))
#     print(s.combinationSum([2, 3, 6, 7], 7))
    print(s.subsets([1, 2, 3]))
