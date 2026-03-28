'''
Idea: 
- 현재 "숫자가 얼마냐"보다 "N을 몇 개 썼느냐"가 더 중요하다. 
- dp[k]는 k개를 사용했을 때 min 이면, min()안에 계산해야할 경우의 수가 너무 많다. 
- dp[k]는 N을 정확히 k개 사용해서 만들 수 있는 숫자들의 집합으로 하면, 
- dp[k] = set() of 
    - concat(k)
    - 모든 a in dp[i] and b in dp[k-i]에 대해 
        - a + b
        - a - b
        - a * b 
        - a // b (b != 0)
    을 전부 넣으면 된다. 

<DP steps>
- overlapping small problem: dp[k]는 N이 k번 사용되었을 때 만들어지는 수 
- state: dp[k], k = N이 사용된 개수 
- what to store: k개만큼 N이 사용되었을 때 나올 수 있는 수들의 set 
- transition: dp[k] = set(concat(k))
    for i in range(1, k): # k개보다 작은 개수가 사용된 모든 (i, k-1) pair에 대해
        # a와 b의 조합 
        for a in dp[i]:
            for b in dp[k-i]:
                set.add(dp[i] + dp[k-i])
                set.add(dp[i] - dp[k-i])
                set.add(dp[i] * dp[k-i])
                set.add(dp[i] // dp[k-i])

- base case: dp[1] = 1 
'''
def solution(N, number):
    def concat(cnt):
        nonlocal N
        return int(str(N) * cnt)
    
    if N == number:
        return 1
    
    # return 값은 8보다 크면 -1을 리턴한다. 
    # 즉, dp table은 dp[8]까지 채우면 된다. 
    dp = [set() for _ in range(9)] # 1~8 사용 
    
    for k in range(1, 9):
        # 현재 개수 k에 대해 
        dp[k].add(concat(k)) # 이어붙인 수: 5, 55, 555 ...

        for i in range(1, k):
            for a in dp[k-i]:
                for b in dp[i]:
                    dp[k].add(a + b)
                    dp[k].add(a - b) 
                    dp[k].add(a * b) 
                    if b != 0:
                        dp[k].add(a // b) 
        if number in dp[k]:
            return k
    return -1

if __name__ == "__main__":
    print(solution(5, 12))