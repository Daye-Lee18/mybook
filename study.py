import sys 

input = sys.stdin.readline 
INF = int(1e9)

# n=노드 개수, m=간선 개수 
n, m = map(int, input().split())
start = int(input())


visited = [False] * (n+1)
# 최단 거리 테이블 무한으로 초기화 
distance = [INF] * (n+1)

# 그래프 입력 받기 
graph = [[] for i in range(n+1)]
for i in range(m):
    # a에서 b노드로 가는 비용이 c 
    a, b, c = map(int, input().split())
    graph[a].append((b, c))


def get_smallest_node():
    min_value = INF 
    index = 0 # 가장 최단 거리가 짧은 노드 (인덱스)
    for i in range(1, n+1):
        if distance[i] < min_value and not visited[i]:
            min_value = distance[i]
            index = i 
    return index 


def dijkstra(start):
    # 시작 노드에 대해서 초기화 
    distance[start]= 0
    visited[start] = True 

    for j in graph[start]:
        distance[j[0]] = j[1] 


    # 시작 노드를 제외한 n-1개의 노드에 대해 반복 
    for i in range(n-1):
        # 현재 최단 거리가 가장 짧은 노드를 꺼내서, 방문 처리 
        now = get_smallest_node()
        visited[now] = True 

        # 현재 노드와 다른 연결된 노드 확인 
        for j in graph[now]:
            cost = distance[now] + j[1]
            
            # 현재 노드를 거쳐서 다른 노드로 이동하는 거리가 더 짧은 경우 
            if cost < distance[j[0]]:
                distance[j[0]] = cost 


# 다이익스트라 알고리즘 수행 
dijkstra(start)

# 모든 노드로 가기 위한 최단 거리 출력 
for i in range(1, n+1):
    # 도달할 수 없는 경우 
    if distance[i] == INF:
        print('INFINITY')
    else:
        print(distance[i])

