import collections
import functools
import re
import sys
from typing import List


class Solution:
    # leetcode.com/problems/two-sum
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        # 브루트 포스
        # for i in range(len(nums)):
        #     for j in range(i+1, len(nums)):
        #         if nums[i] + nums[j] == target:
        #             return [i, j]

        # in을 이용한 탐색
        # for i, n in enumerate(nums):
        #     # print(i, n)
        #     complement = target - n
        #     if complement in nums[i+1:]:
        #         return [nums.index(n), nums[i+1:].index(complement)+(i+1)]

        # 첫 번째 수를 뺀 결과 키 조회
        # 타겟 - 첫 번째 수 = 두 번째 수
        # 두 번째 수를 키 , 인덱스를 값으로 하면 O(1)에 조회 가능
        nums_map = {} # 딕셔너리 생성
        # 딕셔너리에 저장
        for idx, num in enumerate(nums):
            nums_map[num] = idx

        # 타겟에서 첫 번째 수를 뺀 결과를 키로 조회

        for idx, num in enumerate(nums):
            element = target - num
            if element in nums_map and idx != nums_map[element]:
                return [idx, nums_map[element]]

    # leetcode.com/problems/trapping-rain-water
    def trap(self, height: List[int]) -> int:
        # 투포인터 이용
        if not height:
            return 0
        volume = 0
        left, right = 0, len(height)-1
        left_max = height[left]
        right_max = height[right]

        while left < right:
            left_max = max(left_max, height[left])
            right_max = max(right_max, height[right])

            # 더 높은 쪽을 향해 투 포인터 이동
            if left_max <= right_max:
                volume += left_max - height[left]
                # print('left ', volume)
                left += 1
            else:
                volume += right_max - height[right]
                right -= 1
                # print('right ', volume)

        return volume

        # 스택 풀이
        # stack = []
        # volume = 0
        # for i in range(len(height)):
        #     # 변곡점을 만나는 경우
        #     while stack and height[i] > height[stack[-1]]:
        #         # 스택에서 꺼낸다
        #         top = stack.pop()
        #
        #         if not len(stack):
        #             print('not len')
        #             break
        #
        #         # 이전과의 차이만큼 물 높이 처리
        #         distance = i - stack[-1] - 1
        #         waters = min(height[i], height[stack[-1]]) - height[top]
        #
        #         volume += distance * waters
        #     stack.append(i)
        # return volume

    # leetcode.com/problems/3sum
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        results = []
        nums.sort()

        for i in range(len(nums)-2):
            # 중복된 값 건너뛰기
            if i > 0 and nums[i] == nums[i-1]:
                continue

            # 간격을 좁혀가며 합 sum 계산
            left, right = i+1, len(nums) - 1
            while left < right:
                sum = nums[i] + nums[left] + nums[right]
                if sum < 0:
                    left += 1
                elif sum > 0:
                    right -= 1
                else:
                    # sum = 0인 경우이므로 정답 및 스킵처리
                    results.append([nums[i], nums[left], nums[right]])

                    while left < right and nums[left] == nums[left+1]:
                        left += 1
                    while left < right and nums[right] == nums[right-1]:
                        right -= 1
                    left += 1
                    right -= 1

        return results

    # leetcode.com/problems/array-partition-i
    def arrayPairSum(self, nums: List[int]) -> int:

        # 오름차순 풀이
        # nums.sort()
        # result = 0
        # pair = []
        # for i in nums:
        #     pair.append(i)
        #     if len(pair) == 2:
        #         result += min(pair)
        #         print(min(pair))
        #         pair = []
        # return result

        # 파이썬다운 풀이
        # 정렬된 리스트에서 두 개씩 짝지으면 짝수 값이 항상 작은 값이다.
        print(sorted(nums)[::2])
        return sum(sorted(nums)[::2])

    # leetcode.com/problems/product-of-array-except-self
    def productExceptSelf(self, nums: List[int]) -> List[int]:
        result = []
        p = 1
        # 왼쪽 곱셈
        for i in range(0,len(nums)):
            result.append(p)
            p = p * nums[i]
        p = 1

        # 왼쪽 곱셈 결과에 오른쪽 값을 차례대로 곱셈
        for i in range(len(nums)-1, -1, -1):
            result[i] = result[i] * p
            p = p * nums[i]
            # print('p = ', p)
        return result

    # leetcode.com/problems/best-time-to-buy-and-sell-stock
    def maxProfit(self, prices: List[int]) -> int:
        # 브루트포스
        # max_price = 0
        # for i, price in enumerate(prices):
        #     for j in range(i, len(prices)):
        #         max_price = max(prices[j] - price, max_price)
        #
        # return max_price
        # 저점과 현재 값과의 차이 계산
        profit = 0
        min_price = sys.maxsize

        # 최솟값과 최댓값을 계속 갱신
        for price in prices:
            min_price = min(min_price, price)
            profit = max(profit, price-min_price)
        return profit


if __name__ == '__main__':
    s = Solution()
    # print(s.twoSum([2,7,11,15],9))
    # print(s.trap([0,1,0,2,1,0,1,3,2,1,2,1]))
    # print(s.threeSum([-1,0,1,2,-1,-4]))
    # print(s.arrayPairSum([1,4,3,2]))
    # print(s.productExceptSelf([1,2,3,4]))
    print(s.maxProfit([7,1,5,3,6,4]))