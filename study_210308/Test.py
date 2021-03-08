import collections
import functools
import re
import sys
from typing import List


class Solution:
    # leetcode.com/problems/valid-palindrome
    def isPalindrome(self, s: str) -> bool:
        s = s.lower()
        strs = []
        for ch in s:
            # isalnum() -> 영문자와 숫자 판별
            if ch.isalnum():
                strs.append(ch)

        # 문자열 뒤집어서 확인
        return strs == strs[::-1]

    # leetcode.com/problems/reorder-data-in-log-files
    def reorderLogFiles(self, logs: List[str]) -> List[str]:
        # 문자열 로그와 숫자 로그를 구분
        letters = []
        digits = []
        # log 사이즈만큼 반복하면서 isdigit() 메소드를 통해 숫자인지 아닌지 구분
        for log in logs:
            if log.split(' ')[1].isdigit():
                digits.append(log)
                # print(log.split()[1])
            else:
                letters.append(log)
                # print(log)
        # 2개의 키를 람다 표현식으로 정렬
        letters.sort(key=lambda x: (x.split(' ')[1:], x.split(' ')[0]))
        return letters + digits

    # leetcode.com/problems/most-common-word
    def mostCommonWord(self, paragraph: str, banned: List[str]) -> str:
        # 문자열 전처리 -> 콤마 공백 없애기
        # 정규식 사용
        # \w -> Word Character, ^ -> not
        # '[^\w]', ' ' = Word Character 가 아닌 것은 공백으로 바꿈

        words = [word for word in re.sub(r'[^\w]', ' ', paragraph)
                 .lower().split()
                 if word not in banned]

        # Counter 사용
        counts = collections.Counter(words)
        return counts.most_common(1)[0][0]

    # leetcode.com/problems/group-anagrams
    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        anagrams = collections.defaultdict(list)
        for word in strs:
            # 정렬해서 딕셔너리에 추가
            anagrams[''.join(sorted(word))].append(word)
        # print(anagrams.values())
        return list(anagrams.values())

    # leetcode.com/problems/longest-palindromic-substring
    def longestPalindrome(self, s: str) -> str:
        # 팰린드롬 판별 및 투 포인트 확장
        def expand(left: int, right: int) -> str:
            while left >= 0 and right < len(s) and s[left] == s[right]:
                left -= 1
                right += 1
            return s[left+1:right]
        if len(s) < 2 or s == s[::-1]:
            return s
        result = ''
        # 슬라이딩 윈도우 우측으로 이동
        for i in range(len(s) - 1):
            result = max(result, expand(i, i + 1), expand(i, i + 2), key=len)

        return result


if __name__ == '__main__':
    s = Solution()
    print(s.isPalindrome("abba"))
    print(s.reorderLogFiles(["dig1 8 1 5 1", "let1 art can", "dig2 3 6", "let2 own kit dig", "let3 art zero"]))
    print(s.mostCommonWord("Bob hit a ball, the hit BALL flew far after it was hit.", ["hit"]))
    print(s.groupAnagrams(["eat", "tea", "tan", "ate", "nat", "bat"]))
    print(s.longestPalindrome("babad"))