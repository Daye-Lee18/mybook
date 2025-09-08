from collections import deque 

"""
시뮬레이션 + 연결 전파
입력: 숲 크기 RXC, 골렘 K개
1. 골렘은 십자 (+) 모양: 중심 + 상하좌우 1칸 (총 5칸). 출구는 중심에서 바라보는 방향의 이웃 한 칸 
1. RXC를 (R+3)xC로 만들어서, 골렘의 팔까지 생각해서 만든다. gid 하나씩 들어오면, 
2. 맨 아래로 보낸다. (직하, 좌하, 우하). 셋 다 불가면 정지 
3. 만약 맨 아래로 간 4방 팔이 원래 숲 (행 인덱스 3~R+2) 밖에 있으면 전체 초기화 후 해당 골렘은 무시  
4. 만약 골렘의 출구가 인접 골렘과 연결되어 있다면 이동하여 최대 C를 구하고 더한다. 
"""

R, C, K = map(int, input().split())
H = R + 3 

golem_arr = [[0 for _ in range(C)] for _ in range(R+3)] # 골렘의 위치를 gid로 표시 
exit_map = [[0 for _ in range(C)] for _ in range(R+3)] #출구 방향 0~3 표시 
max_row = [0] * (K+1) # K명의 골렘의 max값 채우기 (gid = index 1부터 시작)

# golem의 arm방향 
AY = [-1, 1, 0, 0]; AX = [0, 0, -1, 1]

# 남쪽에서 확인해야할 부분 
DY = [-1, -2, -1]; DX = [-1, 0, 1]

# 좌하에서 확인해야할 부분 5개  
LY = [-1, 1, 2, 0, 1]; LX = [-1, -1, -1. -2, -2] 

# 우하에서 확인해야할 부분 5개 
RY = [-1, 1, 2, 0 ,1]; RX = [1, 1, 1, 2, 2] 

def can_move(y, x, yofs, xofs):
    # 0 ~ H안에 존재하는지 
    for dy, dx in zip(yofs, xofs):
        ny = y + dy
        nx = x + dx 
        # 이동 중에는 골렘의 팔들이 0~H안에만 있고 해당 위치에 다른 골렘들이 없으면 됨.  
        if not (0 <= ny < H and 0 <= nx < C): return False
        if golem_arr[ny][nx] != 0: return False # 비어있지 않으면 False 
    return True # 비어있어야 이동 가능 

def drop_or_rotate(y, x, d):
    # 남쪽으로 최대한 내리기 
    while True:
        # 남쪽 방향으로 이동할때 체크해야할 3부분이 비어있는지 확인 
        if can_move(y, x, DY, DX):
            y +=1 # 남하 
            continue 
        if can_move(y, x, LY, LX):
            y += 1 ; x -= 1 # 좌하
            d =  (d+3)%4
            continue 
        if can_move(y,x, RY, RX):
            y +=1; x += 1  # 우하 
            d = (d+1)%4 
            continue 
        break 
    return y, x, d 

def settle_or_reset(y, x, d, gid):
    for t in range(4):
        ay = y + AY[t]; ax = x + AX[t] 
        if 3 <= ay < H and 0 <= ax < C:
            continue 
        else:
            # reset 
            for a in range(H):
                for b in range(C):
                    golem_arr[a][b] = 0 
            # return False 
            return False 
        
    # settle 
    golem_arr[y][x] = gid  # 가운데 settle 
    exit_map[y + AY[d]][x + AX[d]] = d # 출구 방향 표시 
    for t in range(4): # 4팔 settle 
        ay = y + AY[t]; ax = x + AX[t] 
        golem_arr[ay][ax] = gid
    return True  

def compute_component_max_row(y, x, d, gid):
    # BFS로 주변에 이어져있는 골렘이 있다면 max값 update 
    q = deque([(y, x, d, gid)])

    best = y - 1 
    while q:
        y, x, d, gid = q.popleft()

        # for t in range(8):
        # 출구 방향에 다른 골렘이 있는지 확인 
        ny = y + AY[d]; nx = x + AX[d]
        if ny < 0 or nx < 0 or ny >= H or nx >= C:
            continue 
        if golem_arr[ny][nx] != 0: # 특정 골렘이 위치해있음 
            next_gid = golem_arr[ny][nx]
            q.append((ny, nx, exit_map[ny][nx], next_gid))

            # 업데이트 
            best = max(best, max_row[gid])

    max_row[gid] = max(best, max_row[gid])



        
def solve():
    answer = 0

    # gid는 index로도 활용되기 때문에, 1부터 시작 (golem_arr[y][x] = gid가 0이 아니여야함)
    for gid in range(1, K+1):
        c, d = map(int, input().split())
        y, x = 1, c-1 

        # 최대로 아래로 내림 
        y, x, d = drop_or_rotate(y, x, d)

        # 골렘이 원래 숲의 밖에 있으면 reset 안에 있으면, 골렘의 위치를 settle 
        if not settle_or_reset(y, x, d):
            continue  # reset한 경우에는 무시 

        # 최대 합 구하기 
        compute_component_max_row(y, x, d, gid)
        answer += max_row[gid]

    print(answer)

if __name__ == '__main__':
    solve()
        