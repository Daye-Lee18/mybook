# 5x5중에서 3x3격자 선택 및 회전 CW: 90도, 180도, 270도 
# 향상 회전 : 중심좌표를 기준으로 90도 회전 
'''
### 탐사 진행: 회전 목표 
1)유물 1차 회득 가치 최대화
2) 1)의 방법이 여러개이면, 회전한 각도 중 가장 작은 각도 선택
3) 2)의 방법도 여러가지이면 (회전 중심좌표가 다를 수 있음), 회전 중심 좌표의 열이 가장 작은 구간 선택 
4) 열이 같다면 행이 가장 작은 구간 선택 

### 유물 1차 획득 
- 유물의 가치: 5x5행렬에서 모인 조각의 개수 -> "3개 이상"부터 획득 가능  
- 유물이 사라짐. 
- 새로들어오는 유물은 유적의 벽면에 써 있는 숫자대로 진행 (row up, column up순으로 채워짐)
- 사용된 숫자다음부터 다음에 사용할 수 있음 

#### 유물 연쇄 획득 
- 새로운 유물 조각이 생겨난 이후에도 유물이 있으면 조각을 획득하고 없앤후 다시 채움.
- 다만 더 이상 조각이 3개 이상 연결되지 않아 유물이 될 수 없으면 멈춤 

#### 탐사 반복 
- 탐사 진행 -> 유물 1차 획득 -> 유믈 연쇄 획득 과정까지 1턴이며 총 K번 턴을 돌림. 
- 1번의 turn에서 K번 이전에 어떠한 방법을 사용해서라도 유물을 획득할 수 없다면, 모든 탐사는 그 즉시 종료됨. 
이 경우 얻을 수 있는 유물이 존재하지 않으므로, 종료되는 턴에 아무 값도 출력하지 않음. 
'''

from collections import deque 
from typing import List 

def solve():

    # f = open('/Users/dayelee/Documents/GitHub/mybook/Input.txt', 'r')
    K, M = map(int, input().split())
    global graph, parts
    graph = []

    for n in range(5):
        graph.append(list(map(int, input().split())))

    # 유물조각은 들어온 순서부터 pop
    parts = deque(list(map(int, input().split())))

    
    for k in range(K):
        total = 0

        # Step 1: 
        # 3x3을 회전: 총 9개 위치를 중심으로 CW 90, 180, 270도 (9 * 3=27)개 중 선택, 유물은 조각 3개 이상 연결 
        # 유적위치 locs = list(), 유물의 가치 = len(locs), 
        locs, result_graph = explore()

        if len(locs) == 0:
            break # 유적의 가치가 없으면 K 턴 전에 stop 
        
        total += len(locs) # 유물의 가치 더하기 
        graph = result_graph[:] # 유적 graph update 

        # global graph에 유적 위치 Locs에 새로운 조각 update 
        update_graph(locs)
        
        # global graph에 유물 연쇄 획득 과정 
        value= get_chained_parts()
        total += value 

        # 공백을 사이에 두고 출력 
        print(total, end=' ')


def compare(fy, fx, ry, rx):
    if fx != rx: 
        return fx < rx   # 열 번호가 작은 순 
    elif fy != ry:
        return fy > ry  # 행 번호가 큰 순 
    return True 

def sort_locs(locs: List):
    '''
    ascending order 
    '''
    N = len(locs)
    for f_pointer in range(N):
        lowest_pointer = f_pointer 
        for r_pointer in range(f_pointer+1, N):
            if not compare(locs[lowest_pointer][0], locs[lowest_pointer][1], locs[r_pointer][0], locs[r_pointer][1]):
                # 저장 
                lowest_pointer = r_pointer 
        # swap 
        if lowest_pointer != f_pointer:
            temp = locs[lowest_pointer]
            locs[lowest_pointer] = locs[f_pointer]
            locs[f_pointer] = temp 


def update_graph(locs):
    global graph, parts 

    sort_locs(locs) # call by reference 
    # 정렬 순서대로 update 
    for cy, cx in locs:
        graph[cy][cx] = parts.popleft()



def in_circle(y, x, cy, cx):
    return cy -1 <= y <= cy + 1 and cx -1 <= x <= cx + 1

def explore():
    global graph
    # 열이 가장 작고 -> 행이 가장 작은 순으로 배열  
    # centers = [(1, 1), (1, 2), (1, 3), (2,1), (2,2), (2, 3), (3, 1), (3, 2), (3, 3)]
    centers = [(1, 1), (2, 1), (3, 1), (1, 2), (2, 2), (3, 2), (1, 3), (2, 3), (3, 3)]

    max_value = 0
    max_locs = []
    min_rotation = 99999999 
    result_graph =[[0] * 5 for _ in range(5)]
    for cy, cx in centers:
        local_graph = [row[:] for row in graph]
        previous_local_graph = [row[:] for row in local_graph]
        # 90, 180, 270 CW rotation 
        for rotation_cnt in range(3):
            # 회전 후의 새로운 graph생성 
            add_num = cy + cx 
            x_minus_y = cx - cy 
            for y in range(5):
                for x in range(5):
                    # if (y!=cy and x!=cx) and in_circle(y, x, cy, cx):
                    if not (y==cy and x==cx) and in_circle(y, x, cy, cx):
                        local_graph[y][x] = previous_local_graph[add_num-x][y+x_minus_y] # CW 90도 회전 
                    else:
                        local_graph[y][x] = previous_local_graph[y][x]

            # 이전 rotaed graph update 
            previous_local_graph = [row[:] for row in local_graph]

            # rotation 후 고정된 Graph에서 3개 이상 모여있는 유물의 위치 계산
            cur_locs = calculate_values(previous_local_graph)
            
            # locs, rotation_cnt비교 
            # 각도가 작은 것이 제일 먼저 priority : 각도가 같으면, 열 -> 행 순서대로 이미 적용되어있기 때문에, 가치가 더 클때만 바꾼다. 
            # 따라서 오직 이전것보다 큰 경우에만 update (같으면 앞의 것으로 함.)
            if len(cur_locs) >= max_value: # Step 1: 유물 가치가 가장 높은 것을 최대화 
                if len(cur_locs) == max_value: # Step 2: Step 1이 여러개라면, 회전 각도가 가장 작은 것 
                    if min_rotation > rotation_cnt:
                        # update 
                        min_rotation = rotation_cnt 
                        max_value = len(cur_locs)
                        max_locs = list(cur_locs)
                        result_graph = [row[:] for row in previous_local_graph]
                else:
                    # update 
                    min_rotation = rotation_cnt 
                    max_value = len(cur_locs)
                    max_locs = list(cur_locs)
                    result_graph = [row[:] for row in previous_local_graph]

    return max_locs,result_graph

def in_range(y, x):
    return 0<=y<5 and 0<=x <5 

def BFS(y, x, cur_graph):
    q = deque([(y, x)])
    visited = set()
    visited.add((y, x))
    DY = [-1, 1, 0, 0]; DX = [0, 0, 1, -1]

    while q:
        cur_y, cur_x = q.popleft()

        for t in range(4):
            ny = cur_y + DY[t]; nx = cur_x + DX[t] 
            # range안에 있고 & 원래 y, x 안에 있는 수와 옆의 수가 동일한지 & 방문하지는 않았는지 
            if in_range(ny, nx) and cur_graph[ny][nx] == cur_graph[y][x] and (not (ny, nx) in visited):
                q.append((ny, nx))
                visited.add((ny,nx))

    # 3개 이상인지 
    if len(visited) >= 3:
        return True, visited
    else :
        return False, None 

def calculate_values(cur_graph):
    '''
    고정된 Graph에서 3개 이상 모여있는 유물의 위치 계산
    '''
    result = set()

    for y in range(5):
        for x in range(5):
            is_more_three, locs = BFS(y, x, cur_graph)
            if is_more_three:
                # set을 extend하는 방법? 
                result = result.union(locs)
                    
    return result 
            

def get_chained_parts():
    global graph 
    values = 0 
    while True:
        # 유적 위치 세기 
        locs = calculate_values(graph)

        if len(locs) == 0:
            break 
 
        values += len(locs) 

        # 유적 graph 업데이트하기 
        update_graph(list(locs))
    return values


if __name__ == '__main__':
    solve()