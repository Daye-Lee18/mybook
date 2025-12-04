import sys 

sys.stdin = open('Input.txt', 'r')

from collections import deque 

R, C, K = map(int, input().split())
H = R + 3 
graph = [[0] * C for _ in range(H)]
id_to_center_dir = dict()

'''
숲의 바깥방향에서 시작해 골렘의 중아이 c열이 되도록 하는 위치에서 내려옴. 
1) 남쪽 2) 서쪽 rotation + 아래 한 칸 3) 동쪽 rotation + 아래 한칸 
---> 가장 남쪽 도달 후에는 정령은 골렘 내에서 상하좌우 인접칸으로 이동 가능 
---> 최종에서 골렘의 몸 일부가 여전히 숲을 벗어난 상태라면 모든 골렘 삭제 및 새롭게 숲  탐색 -> 이 경우 최종 위치를 답에 포함시키지 않음. 
---> 골렘의 출구가 다른 골렘과 인접하다면 해당 출구를 통해 다른 골렘으로 이동 가능 
----> 정령이 도달하게 되는 최종 위치 누적 
정령은 어떤 방향에서든 골렘에 탑승 가능, 내릴 때에는 정해진 출구를 통해서만 내릴 수 있음. 
'''
# 북, 동, 남 ,서 
AY = [-1, 0, 1, 0]; AX = [0, 1, 0, -1]

DY = [1, 2, 1]; DX = [-1, 0, 1]
LY = [-1, 0, 1, 1, 2]; LX = [-1, -2, -2, -1,-1]
RY = [-1, 0, 1 ,1, 2]; RX = [1, 2, 1, 2, 1]

max_values = [0] * (K+1)

def in_range(y, x):
    return 0<=y <H and 0<=x < C 

def in_forest(y, x):
    return 3 <= y < H and 0<=x < C

def can_move(cur_y, cur_x, DIR_Y, DIR_X):
    for dt_y, dt_x in zip(DIR_Y, DIR_X):
        ny = cur_y + dt_y 
        nx = cur_x + dt_x 
        if not in_range(ny, nx) or graph[ny][nx] != 0:
            return False
    
    return True 

def place(center_y, center_x, dir):
    y = center_y; x = center_x
    
    while True:
        if can_move(y, x, DY, DX): # 아래 
            y += 1 
            continue 

        if can_move(y, x, LY, LX): # 왼쪽 아래 
            y += 1; x -= 1 
            dir = (dir +3)%4
            continue 
         
        if can_move(y, x, RY, RX): # 오른쪽 아래 
            y += 1; x += 1 
            dir = (dir+1)%4
            continue 

        # 위의 3가지 경우로 못가는 경우 멈춤 
        break 

    return y, x, dir



def reset_or_settle(cy, cx, dir, spirit_id):
    global id_to_center_dir
    # reset 
    for t in range(4):
        ny = cy + AY[t]; nx = cx + AX[t]
        if not in_forest(ny, nx):
            # reset :graph and id_to_center_dir
            for y in range(H):
                for x in range(C):
                    graph[y][x] = 0 

            id_to_center_dir = dict()
            return False 
    # settle 
    graph[cy][cx] = spirit_id
    for t in range(4):
        ay = cy + AY[t]; ax = cx + AX[t]
        graph[ay][ax] = spirit_id

    # id_to_center_dict update 
    id_to_center_dir[spirit_id] = (cy, cx, dir)
    return True 
    
def exit_cell(id_):
    cy, cx, cd = id_to_center_dir[id_]
    return cy + AY[cd], cx + AX[cd]

def calculate(id):
    # 현재 위치 Max값 구하기 
    cy = id_to_center_dir[id][0]
    cx = id_to_center_dir[id][1]
    c_dir = id_to_center_dir[id][2]

    # exit 
    ey = cy + AY[c_dir] 
    ex = cx + AX[c_dir]
    
    best = cy + 1 - 2 # 현재 위치에서 골렘 아래 팔로 내려가는 것 - 2(원래 map에서의 column값)이 가장 큰 값 
    # 현재 골렘의 출구와 연결되어 있는 골렘들의 max값으로 update 
    # 이미 다른 골렘들은 연결되어 있는 골렘들이 가진 max값으로 update되어 있기 때문에 출구에서 4 방향을 보는 것만으로 충분함. 
    for t in range(4):
        ny = ey + AY[t]
        nx = ex + AX[t]
        if in_forest(ny, nx) and graph[ny][nx] != 0 and graph[ny][nx] != id:
            best = max(best, max_values[graph[ny][nx]])

    # 현재 H 행이므로 H = R + 3 인데, 첫 행을 1부터 시작하므로 -3 + 1 해서 -2 
    max_values[id] = best

    # propagate 
    # 현재 골렘 옆에 있는 골렘 값들도 max 값으로 update해주기
    # 8방향에서 내 쪽으로 들어올 수 있기 때문에, "옆에 출구가 있는 것들에 한해" 업데이트 해주기?! 
    # 연결되어 있지 않은 골렘들이 현재 id에 내려온 골렘때문에 연결될 수 있으므로 propagation 
    Neighbor_Y = [-2, -1, -1, 0, 0, 1, 1, 2]
    Neighbor_X = [0, -1, 1, -2, 2, -1, 1, 0]

    # BFS 
    q = deque([(cy, cx, id)]) # center y, x값을 넣어야함. 
    visited = set()
    visited.add(id)
    while q:
        y, x, cur_id = q.popleft()

        for t in range(8):
            ny = y + Neighbor_Y[t]; nx = x + Neighbor_X[t] 
            if in_forest(ny, nx) and graph[ny][nx] != 0 and graph[ny][nx] != cur_id:
                # 현재 위치가 다른 골렘의 출구인 경우만, propgate 
                if exit_cell(graph[ny][nx]) == (ny, nx):
                    n_id = graph[ny][nx]
                    if not n_id in visited:
                        visited.add(n_id)
                        # update 
                        max_values[n_id] = max(max_values[n_id], best)
                        q.append((id_to_center_dir[n_id][0], id_to_center_dir[n_id][1], n_id)) # center 값 update 

def solve():
    total = 0
    for spirit_id in range(1, K+1):
        
        c, d = map(int, input().split())
        # print(f"from {c-1} column")
        y = 1 ; x = c -1 
        # place, # 현재 골렘안의 정령이 최대로 갈 수 있는 위치 구하기 
        cur_y, cur_x, cur_dir = place(y, x, d)

        # forest밖이면 reset, 아니면 graph및 id_to_center_dirs에 표시 
        if not reset_or_settle(cur_y, cur_x, cur_dir, spirit_id):
            # print(f"max y of {spirit_id}: ignore")
            # for row in graph:
            #     print(row[:])
            continue 

        calculate(spirit_id)
        # print(f"max y of {spirit_id}: {max_values[spirit_id]}")
        # for row in graph:
        #     print(row[:])
        total += max_values[spirit_id]
    
    print(total)

if __name__ == "__main__":
    solve()