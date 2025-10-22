# p4.py — Min Cost Climbing Stairs (DP)

from typing import List

class Solution:
    def minCostClimbingStairs(self, cost: List[int]) -> int:
        n = len(cost)
        if n == 0: return 0
        if n == 1: return 0  # 시작을 0 또는 1에서 할 수 있으므로, 한 칸만 있으면 0
        dp0, dp1 = 0, 0  # 도착 전 두 계단의 최소비용 (i-2, i-1에서 i로)
        for i in range(2, n + 1):
            dpi = min(dp1 + cost[i-1], dp0 + cost[i-2])
            dp0, dp1 = dp1, dpi
        return dp1


def run_tests():
    sol = Solution()
    tests = [
        ([10,15,20], 15),
        ([1,100,1,1,1,100,1,1,100,1], 6),
        ([0,0,0,0], 0),
        ([5,4,3], 4),
        ([1,2], 1),
        ([0,1,1,0], 1),
        ([2,2,2,2], 4),
        ([9,1,1,9,1], 3),
        ([3,0,2,0,4], 0),
        ([1,1,1], 1),               # 10
        ([10,1,10,1], 2),           # 11
        ([1,100,100,1], 101),       # 12
        ([0,5], 0),                 # 13
        ([5,0], 0),                 # 14
        ([1,0,0,1,0], 0),           # 15
        ([8,9,10,1,1,1], 11),       # 16
        ([100,1,1,100], 2),         # 17
        ([7,7,7,7,7,7], 21),        # 18
        ([0,0,0,0,0,10], 0),        # 19
        ([5,10,5,10,5], 15),        # 20
    ]
    passed = 0
    for i, (inp, expected) in enumerate(tests, 1):
        got = sol.minCostClimbingStairs(inp)
        ok = got == expected
        print(f"[p4][case {i}] cost={inp} -> {got} (expected {expected}) {'OK' if ok else 'FAIL'}")
        passed += ok
    print(f"[p4] Passed {passed}/{len(tests)}")


if __name__ == "__main__":
    run_tests()
