import sys 
from collections import deque 

N = int(input())

tree = [[] for _ in range(N+1)]
for _ in range(N-1):
    u, v = map(int, input().split())
    tree[u].append(v)
    tree[v].append(u)

dp = [[0, 1] for _ in range(N+1)]

def solve():
    visited = [False] * (N+1)
    start = 1 
    que = deque([start])
    stack = []

    while que:
        parent = que.popleft()
        visited[parent] = True 

        for child in tree[parent]:
            if not visited[child]:
                que.append(child)
                stack.append([parent, child])
    
    while stack:
        parent, child = stack.pop()
        dp[parent][0] += dp[child][1]
        dp[parent][1] += min(dp[child])

solve()
print(min(dp[1]))