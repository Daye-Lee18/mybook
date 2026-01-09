from typing import List 
from bisect import bisect_right, bisect_left 
from collections import defaultdict 
'''
You may assume that each input would have exactly one solution, 
and you may not use the same element twice.
'''
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        n = len(nums)

        index_dict = defaultdict(list)
        for idx, num in enumerate(nums):
            index_dict[num].append(idx)

        nums.sort()
        for l_index in range(n):
            l = bisect_left(nums, target - nums[l_index])
            r = bisect_right(nums, target-nums[l_index])

            if l == n: # 해당 숫자가 없음 
                continue 

            cnt = r - l # 필요한 숫자의 개수 

            if cnt == 1:
                return [index_dict[nums[l_index]][0], index_dict[nums[l]][0]]
            elif cnt >= 2: # cnt >= 2 
                return [index_dict[nums[l_index]][0], index_dict[nums[l_index]][1]]


nums = [5,75,25]; target = 100 # [1, 2]
# nums = [0,4,3,0]; target = 0 # [0, 3]
# nums = [2, 7, 11, 15]; target = 26 # [2,3]
# nums = [3, 2, 4]; target = 6 # [1, 2]
# nums = [3, 3]; target = 6 # [0, 1]
sol = Solution()
print(sol.twoSum(nums, target))
