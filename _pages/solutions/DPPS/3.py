'''
사람-노드, 에지-친구 
- SNS에 속한 사람은 얼리 어답터/얼리 어답터가 아니다. 
- 가능한 한 최소의 수의 얼리 어답터를 확보하여 '모든 사람이' 이 아이디어를 받아들이게 하기 위한 "최소한의" 얼리 어답터의 수를 구하라.  
- SNS가 트리임을 가정 (graph without a cycle)

constraints 
- N: 정점의 수 2 <= N <= 1e6 , 1-indexed 
- 엣지수: N-1 

'''
import sys 
sys.setrecursionlimit(1_000_001)
input = sys.stdin.readline 

# sys.stdin = open('Input.txt')
N = int(input())

graph = [[] for _ in range(N+1)]
for _ in range(N-1):
    s, d = map(int, input().split())
    graph[s].append(d)
    graph[d].append(s)

# dp[node][0] = 현재 node가 early adopter가 아닐때, 필요한 최소한의 얼리어답터 수 
# dp[node][1] = 현재 node가 early adopter일 때, 필요한 최소한의 얼리어답터 수 
dp = [[0, 1] for _ in range(N+1)] 
visited= [False]*(N+1)

def dfs(node):
    visited[node] = True 

    for child in graph[node]:
        if visited[child]:
            continue 

        dfs(child)
        dp[node][0] += dp[child][1] # subtree의 root가 얼리어답터가 아닌 경우, child nodes들이 다 얼리어답터여야함. 
        dp[node][1] += min(dp[child]) # subtree의 root가 얼리어답터인 경우, 모든 경우의 수인 경우에서 가능하므로, 가장 작은 것을 더해주면 됨. 
    
dfs(1)
print(min(dp[1]))
# print(dp)