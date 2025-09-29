import heapq 
import sys 
input = sys.stdin.readline 
INF = int(1e9)

# 노드의 개수, 간선의 개수를 입력받기 
n, m = map(int, input().split())
# 시작 노드 번호 입력 받기 
start = int(input())
 
# 각 노드에 연결되어 있는 노드에 대한 정보를 담는 리스트 만들기 
graph = [[] for i in range(n+1)]
# 최단 거리 테이블을 모두 무한으로 초기화 
distance = [INF] * (n+1)

# 모든 간선 정보 입력받기 
for _ in range(m):
    a, b, c= map(int, input().split())
    # a번 노드에서 b번 노드로 가는 비용이 c 
    graph[a].append((b, c))

def dijkstra(start):
    q = [] 

    distance[start] = 0
    heapq.heappush(q, (0, start))

    while q:
        cur_dis, node = heapq.heappop(q)

        if cur_dis > distance[node]:
            continue 

        for weight, nxt_node in graph[node]:
            if cur_dis + weight < distance[nxt_node]:
                distance[nxt_node] = cur_dis + weight # 현재 node까지 온 비용 + nxt_node로 가는 비용 
                heapq.heappush(q, (distance[nxt_node], nxt_node))

# 다익스트라 알고리즘 수행 
dijkstra(start)

# 모든 노드로 가기 위한 최단 거리 출력 
for i in range(1, n+1):
    if distance[i] == INF:
        print("INF")
    else:
        print(distance[i])