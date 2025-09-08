from collections import deque

# 0=북, 1= 동, 2=남, 3=서
DY = [-1, 0, 1, 0]
DX = [0, 1, 0, -1]

# 아래 직하/좌하/우하 이동시, 비어있어야하는 상대 좌표 집합
# 중심 기준 상대 좌표들이 모두 비어있어야 해당 ㅣㅇ동 가능
# 아래로: 중심 아래 2칸, 좌/우 아래 1칸
TY = [2, 1, 1]; TX = [0, -1, 1]
# 좌하: 왼쪽으로 틀며 내려갈 때 필요한 빈칸 (모서리 간섭 방지)
LY = [0, 1, -1, 1, 2]; LX = [-2, -1, -1, -2, -1]
# 우하 : 오른쪽으로 틀며 내려갈 때 필요한 빈칸 (모서리 간섭 방지)
RY = [-1, 1, 0, 2, 1]; RX = [1, 1, 2, 1, 2]

# 출구 접속을 전파할 때, 한 골렘 중심 추변에서 "출구 셀"을 탐색할 8방 이웃
EY = [2, -2, 0, 0, -1, -1, 1, 1]
EX = [0, 0, 2, -2, 1, -1, -1, 1]

R, C, K = map(int, input().split())
H = R + 3 # 시뮬은 R위에 3행 더해서 (R+3)xC보드 사용

# arr: 골렘 몸 (심자 5칸)의 골렘 ID 기록 (0=빈칸)
# exit_map = 출구 칸에 골렘 id 기록 (0=없음)
arr = [[0]*C for _ in range(H)]
exit_map = [[0]*C for _ in range(H)]

# 각 골렘 정보와 결과 합
units = [None]*(K+1) # (x,y,dir) 저장용
max_row = [0] * (K+1) # 각 골렘 컴포넌트의 "도달 가능한 최대 행" (원래 숲 기준 0 ~ R-1)
answer = 0

def in_range(y, x):
    """골렘이 움직일 수 있는 구간"""
    return 0<= y< H and 0 <= x <C

def in_forest(y ,x):
    # 원래 숲 범위: 행 3...H-1 (R+2)
    return 3 <= y < H and 0 <= x <C

def can_move(y, x, yofs, xofs):
    """
    직하, 좌하, 우하 중 골렘이 갈 수 있음을 확인
    """
    for dy, dx in zip(yofs, xofs):
        ny = y + dy
        nx = x + dx

        if not in_range(ny, nx) or arr[ny][nx] != 0:
            return False

    return True

def drop_and_settle(y, x, d):
    # x, y,d에서 아래/좌하/우하를 우선순위대로 반복 시도
    while True:
        if can_move(y, x, TY, TX):
            y += 1 # 아래
            continue
        if can_move(y, x, LY, LX):
            d = (d + 3) % 4 # 좌하: 출구 좌회전
            y += 1; x -= 1
            continue
        if can_move(y, x, RY, RX):
            d = (d+1) % 4 # 우하: 출구 우회전
            y += 1; x += 1
            continue
        break
    return y, x, d

def place_or_reset(id_, y, x, d):
    # 십자 네 팔이 원래 숲 범위를 벗어나면 전체 초기화 후 False return
    for dir4 in range(4):
        # arm_y, arm_x
        ay, ax = y + DY[dir4], x + DX[dir4]
        if not in_forest(ay, ax):
            # 전체 리셋
            for i in range(H):
                for j in range(C):
                    arr[i][j] = 0
                    exit_map[i][j] = 0
            return False

    # 십자 네 팔이 원래 숲 범위 안에 있는 경우 True return
    # 출구 표시
    ey, ex = y + DY[d], x + DX[d]
    exit_map[ey][ex] = id_

    # 몸통 (십자) 표시
    arr[y][x] = id_
    for dir4 in range(4):
        ay, ax = y + DY[dir4], x + DX[dir4]
        arr[ay][ax] = id_

    units[id_] = (y, x, d)
    return True

def compute_component_maxrow(start_id):
    """
    현재 골렘 기준으로 최대행 계산 및 출구-접속 가능한 골렘들에 대해
    도달 가능한 최대 행 (원래 숲 좌표계) 값을 전파 (BFS)하며 갱신
    """
    # 기본: 중심의 '숲 내' 행 인덱스 = x-3
    sy, sx, sd = units[start_id]
    # max_row: “사람이 실제로 밟을 수 있는 빈 칸 중 가장 아래(=가장 큰 y)”를 저장
    base = sy - 1 # 골렘의 중심 기준으로 도달 가능한 최대 행을 저장


    # 먼저, 현재 골렘의 출구 주변 4방에 인접한 '몸'이 있으면 그 골렘으로부터 더 큰 최대값을 물려받음
    best = sy - 1 # 중심 칸은 본인이 점유해서 사람이 설 수 없기 때문. 가장 아래쪽의 빈 칸이 중심 바로 위(sy-1)이므로, 거기서부터 최대행을 잡아 전파(BFS)합니다.
    """
    이후 출구 접속으로 다른 골렘과 이어지면, 이 base와 이웃 골렘들의 max_row를 비교해가며 더 큰 값으로 갱신하죠.
    """
    ey, ex = sy + DY[sd], sx + DX[sd]

    for t in range(4):
        ny, nx = ey + DY[t], ex + DX[t]
        if in_range(ny, nx) and arr[ny][nx] != 0 and arr[ny][nx] != start_id:
            nid = arr[ny][nx]
            best = max(best, max_row[nid])

    # 시작 골렘의 현재 best 기록
    if best > max_row[start_id]:
        max_row[start_id] = best

    # 이제 출구가 '맞닿아 있는' 골렘들로 최대 행을 전파 (출구 맵 기준 8방)
    q = deque([start_id])
    seen = {start_id} # set
    while q:
        gid = q.popleft()
        gy, gx, gd = units[gid]

        # 중심 8방의 출구 칸들을 보며, 출구가 있는 다른 골렘을 찾는다.
        for k in range(8):
             ny, nx = gy + EY[k], gx + EX[k]
             if in_range(ny, nx) and exit_map[ny][nx] != 0:
                 # next id
                 nid = exit_map[ny][nx]
                 if nid not in seen:
                     seen.add(nid)
                     # 더 큰 최대행을 전파
                     if max_row[gid] > max_row[nid]:
                         max_row[nid] = max_row[gid]
                     q.append(nid)

def solve():
    global answer
    for gid in range(1, K+1):
        c, d = map(int, input().split())
        # 시작 지점: (1, c-1)에서 낙하 시작 (보드 상단 여유행 고려)
        y, x, dir_ = 1, c-1, d
        y, x, dir_ = drop_and_settle(y, x, dir_)

        # 놓을 수 없으면 전체 보드 초기화되고, 이 골렘은 스킵
        if not place_or_reset(gid, y, x, dir_):
            continue

        # 이 골렘 기준으로 최대행 계산 및 출구 접속 전파
        compute_component_maxrow(gid)
        answer += max_row[gid]

    print(answer)

if __name__ == "__main__":
    solve()