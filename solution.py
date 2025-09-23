from typing import List 
# import bisect 

class Solution:
    def bisect_left(self, arr, x):
        l, r = 0, len(arr)

        while l < r:
            mid = (l+r)//2

            if x <= arr[mid]:
                r = mid 
            else:
                l = mid + 1 

        return l

    def lengthOfLIS(self, nums: List[int]) -> int:
        tails = [] # at index i, the smallest value that has 'i+1' length of a subsequence

        for num in nums:
            idx = self.bisect_left(tails, num)
            if idx == len(tails):
                tails.append(num)
            else:
                tails[idx] = num 
        
        return len(tails)
        