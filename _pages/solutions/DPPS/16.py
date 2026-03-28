'''
overlapping problem: dp[row][col] = root부터 해당 노드까지 path의 최댓값 
state: [row][col]
what to store: root부터 해당 노드까지 path 합의 최댓값
base case: dp[0][0] = triangle[0][0] # root 값 

'''

def solution(triangle):
    # INIT DP table 
    dp = []
    for i in range(len(triangle)):
        dp.append([])
        for j in range(len(triangle[i])):
            dp[i].append(0)

    # base case 
    dp[0][0] = triangle[0][0]

    for cur_h in range(1, len(triangle)):
        for cur_c in range(len(triangle[cur_h])):
            # 현재 index i -> 왼쪽 위, 오른쪽 위
            if cur_c -1 >= 0 and cur_c < len(dp[cur_h-1]):   
                dp[cur_h][cur_c] = max(dp[cur_h-1][cur_c], dp[cur_h-1][cur_c-1]) + triangle[cur_h][cur_c]
            elif cur_c -1 >= 0: # 왼쪽 위만 있는 경우 
                dp[cur_h][cur_c] = dp[cur_h-1][cur_c-1] + triangle[cur_h][cur_c]
            else: # 오른쪽 위만 있는 경우 
                dp[cur_h][cur_c] = dp[cur_h-1][cur_c] + triangle[cur_h][cur_c]


    return max(dp[len(triangle)-1])

if __name__ == "__main__":
    tri= [[7], [3, 8], [8, 1, 0], [2, 7, 4, 4], [4, 5, 2, 6, 5]] # 30
    print(solution(tri)) 