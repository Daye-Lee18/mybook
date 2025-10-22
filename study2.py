import sys
sys.stdin = open('Input.txt', 'r')

n = int(input())
m = int(input())
INF = int(1e9)
dp = [[INF]*(n+1) for _ in range(n+1)]

for i in range(1, n):
    dp[i][i] = 0 

for _ in range(m):
    a, b, c= map(int, input().split())
    dp[a][b] = c 

def floyd():
    for k in range(1, n+1):
        for a in range(1, n+1):
            for b in range(1, n+1):
                dp[a][b] = min(dp[a][b], dp[a][k] + dp[k][b])



floyd()
# print 
for row in dp[1:]:
    for num in row[1:]:
        if num == INF:
            print(0, end=' ')
        else:
            print(num, end=' ')
    print()


