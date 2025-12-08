from collections import deque 

'''
1-indexed (1~N)나라 
- N-1개 길 (edge): tree
- 방향성 없음 

우수마을 선정 
- 우수 마을로 선정된 마을 주민 수의 총 합을 최대로 함 
- 우수 마을끼리는 서로 인접하지 않음 
- 우수 마을로 선정되지 못한 마을은 적어도 '하나의' 우수 마을과는 인접해 있다. 

constraints 
1 <= N <= 1e4 
마을주민수 <= 1e4 

- 마을끼리 그룹을 이루어서 떨어져있다면? 
-> 불가능. 문제에서 마을과 마을사이를 직접 잇는 N-1개의 길이 있다고 하였음. 즉, 이 문제는 트리 문제이고, 떨어져있는 마을들이 없다고 가정함. 
-> cycle도 없음. N개의 마을을 N-1개의 선으로 이을때 cycle을 만들 수 없음. 

문제를 읽어보면, 작은 subtree를 먼저 계산해야 최종적으로 큰 트리에서 최대 합을 구할 수 있으므로, DFS (top-down)방식을 이용해 문제를 푼다. 
'''
import sys 

# sys.stdin = open('Input.txt')
sys.setrecursionlimit(int(1e4)) # 마을의 개수가 최대 1e4개 이므로 재귀를 N까지 올려줌. 

N = int(input())

populations = [0]
populations.extend(list(map(int, input().split())))
visited = [False] * (N+1)

graph = [[] for _ in range(N+1)]
for _ in range(N-1):
    u, v = map(int, input().split())
    graph[u].append(v)
    graph[v].append(u)

# status: each node 
# (현재 노드가 우수노드, 현재 노드가 우수노드아님.)
dp = [[populations[_], 0] for _ in range(N+1)]


def dfs(node):
    global dp 
    visited[node] = True 

    for child in graph[node]:
        if visited[child]:
            continue 

        dfs(child)
        dp[node][0] += dp[child][1]
        dp[node][1] += max(dp[child][0], dp[child][1])

dfs(1) # 아무 노드나 root로 선정 
print(max(dp[1][0], dp[1][1]))
