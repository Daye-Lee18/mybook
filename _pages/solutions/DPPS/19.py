'''
인접한 두 집을 털면 경보 
money: 각 집에 있는 돈이 담긴 배열 money 가 주어진다. 
return: 도둑이 훔칠 수 있는 돈의 최댓값 

- constraints:
    - 집의 개수 3<= <= 1e6
    - 0 <= money elements < 1e3 

- "동그랗게 배치되어 있는 집": cycle 
- XXX: 시작과 end가 연결되어있음에 주의 (0과 len(moeny)-1는 연결)
- linear DP 이지만, cycle이 있음에 유의 
    - cycle을 다룰 땐, 항상 % (mod)를 사용한다.
    - XXX: %로 사이클을 직접 처리하려고 하면 안된다. 

- dp[시작] = [해당 노드는 안전할때 max, 해당 노드가 털릴 때 max]
- transition:
    dp[현재노드][0] = max(dp[이전집][털림], dp[이전집][안털림]) + 현재 집에 있는 돈 
    dp[현재노드][1] = dp[이전집][안털림] + 현재 집에 있는 돈 
- base case:
    dp[시작] = [0, money[시작]]
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