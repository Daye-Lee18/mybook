'''
보통은 최단 경로를 물어보는데, 개수를 물어보네요. 해당 경로까지의 최단 경로를 append로 정리하여 
[1][1]까지 도착하는 경로들의 cost가 a= [3, 4, 5, 3]이면 답은 a.count(min(a)) 가 될 수 있을 듯. 

overlapping subproblems: dp[col][row] = 해당 지점까지 도착하는 경로들의 최소 합의 집합. 
보통은 
'''

def solution(m, n, puddles):
    # 1-indexed (1,1) ~ (m, n) # 주의! col을 앞에 작성 
    # 오른쪽과 아래쪽으로만 움직여 집에서 학교까지 갈 수 있는 최단 경로의 개수 
    # start = (1, 1)
    # target = (m, n)

    # INIT DP
    # 주의! dp[m][n]에는 [최단 경로 weight, 개수] 리스트가 들어있음 
    dp = [[[1, 1] for _ in range(n+1)] for _ in range(m+1)]
    INF_MAX = int(1e9)
    for puddle in puddles:
        dp[puddle[0]][puddle[1]] = [INF_MAX, 0] 

    for cur_col in range(1, m+1):
        for cur_row in range(1, n+1):
            # 해당 지점까지 "왼쪽" 혹은 "위에서" 온다. 
            # 왼쪽 [cur_col-1][cur_row]
            # 위 [cur_col][cur_row-1]
            # paddle인 경우 pass 
            if dp[cur_col][cur_row] == [INF_MAX, 0]:
                continue 
            if cur_row - 1>= 1 and cur_col-1>=1: # 시작은 1,1부터임! 
                lowest_cost = min(dp[cur_col-1][cur_row][0], dp[cur_col][cur_row-1][0])
                # if lowest_cost == dp[cur_col-1][cur_row][0] == dp[cur_col][cur_row-1][0]:
                if (lowest_cost == dp[cur_col-1][cur_row][0]) and (lowest_cost == dp[cur_col][cur_row-1][0]):
                    dp[cur_col][cur_row] = [lowest_cost + 1, dp[cur_col-1][cur_row][1] + dp[cur_col][cur_row-1][1]] 
                elif lowest_cost == dp[cur_col-1][cur_row][0]: 
                    dp[cur_col][cur_row] = [lowest_cost + 1,  dp[cur_col-1][cur_row][1]]
                else:
                    dp[cur_col][cur_row] = [lowest_cost + 1,  dp[cur_col][cur_row-1][1]]
            elif cur_row - 1>= 1: # 왼쪽에서 오는 경우만 있는 경우 
                dp[cur_col][cur_row] = [dp[cur_col][cur_row-1][0] + 1, dp[cur_col][cur_row-1][1]]
            else: # 위에서 오는 경우만 있는 경우 
                dp[cur_col][cur_row] = [dp[cur_col-1][cur_row][0] + 1, dp[cur_col-1][cur_row][1]]
    return dp[m][n][1] % 1_000_000_007


if __name__ == "__main__":
    print(solution(4, 3, [[2, 2]]))# 4