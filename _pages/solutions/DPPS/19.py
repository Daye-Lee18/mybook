'''
이 문제는 "동그랗게 배치된 집 (cycle)"에서 인접한 집을 털 수 없는 문제이다.

하지만 사이클을 직접 처리하기보다, 다음과 같이 두 가지 경우로 나누어 해결한다:

1. 첫 번째 집을 턴 경우
   → 마지막 집은 털 수 없음
   → money[0 : n-1] 구간에서 선형 DP 수행

2. 첫 번째 집을 털지 않은 경우
   → 마지막 집을 털 수 있음
   → money[1 : n] 구간에서 선형 DP 수행

이렇게 두 경우를 각각 계산한 뒤, 최댓값을 선택한다.

---

[선형 도둑질 문제 (House Robber)]

- state:
    dp[i] = i번째 집까지 고려했을 때 훔칠 수 있는 최대 금액

- transition:
    dp[i] = max(
        dp[i-1],              # 현재 집을 털지 않음
        dp[i-2] + money[i]    # 현재 집을 털음
    )

- base case:
    dp[0] = money[0]
    dp[1] = max(money[0], money[1])

---

[핵심 포인트]

- 이 문제는 "사이클 DP"처럼 % 연산으로 처리하는 문제가 아님
- 핵심은 "첫 집과 마지막 집이 동시에 선택될 수 없다"는 제약
- 따라서 두 개의 선형 문제로 분리하는 것이 정답 접근
'''

def solution(money):
    def rob_linear(nums):
        n = len(nums)
        if n == 1:
            return nums[0]
        if n == 2:
            return max(nums[0], nums[1])

        dp = [0] * n
        dp[0] = nums[0]
        dp[1] = max(nums[0], nums[1])

        for i in range(2, n):
            dp[i] = max(dp[i - 1], dp[i - 2] + nums[i])

        return dp[-1]

    # 경우 1: 첫 집 포함, 마지막 집 제외
    case1 = rob_linear(money[:-1])

    # 경우 2: 첫 집 제외, 마지막 집 포함 가능
    case2 = rob_linear(money[1:])

    return max(case1, case2)

if __name__ == "__main__":
    money = [1, 2, 3, 1] # 4 
    print(solution(money))