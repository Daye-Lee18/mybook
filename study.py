

# N = the number of nodes, M = the number of edges 
N, M = map(int, input().split())
INF = int(1e9)
dp = [[INF] * (N+1) for _ in range(N+1)]

# print(dp)

# diagonal of dp table 
for i in range(1, N+1):
    dp[i][i] = 0

# initialize the dp with edges
for edge in range(M):
    # from a to b with c cost 
    a, b, c = map(int, input().split())
    dp[a][b] = c 

# iterate through nodes for an intermidiate node 
for intermidiate in range(1, N+1):
    # update dp 
    for a in range(1, N+1):
        for b in range(1, N+1):
            '''
            if a == k, a->a->b이면, a->a = 0이므로 동일해서 
            if a != k라는 조건이 있을 필요가 없음.
            '''
            dp[a][b] = min(dp[a][b], dp[a][intermidiate] + dp[intermidiate][b])


# print out the dp without node 0
for i in range(1, N+1):
    print(dp[i][1:])