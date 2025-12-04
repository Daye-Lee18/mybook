from collections import deque 
import sys 

# sys.stdin = open("Input.txt")

class Robot:
    def __init__(self, num:int, r:int, c: int):
        self.num = num
        self.r = r 
        self.c = c 


N, K, L = map(int, input().split())
A = [list(map(int, input().split())) for _ in range(N)]
B = [[-1]*N for _ in range(N)]
robots = []

for i in range(K):
    r, c = map(int, input().split())
    rb = Robot(i, r-1, c-1)
    robots.append(rb)
    B[rb.r][rb.c] = rb.num

# 4방(좌,상,우,하) + (0,0) 유지(청소 시 자기 칸 포함용)
dxys = [(0, -1), (-1, 0), (0, 1), (1,0), (0, 0)]

def in_range(x: int, y: int) -> bool:
    return 0 <= x < N and 0 <= y < N

def move(rb: Robot) -> None:
    if A[rb.r][rb.c] > 0:
        return 
    
    nearest = None 
    best_D = -1 

    q = deque()
    dist = [[-1] * N for _ in range(N)]
    q.append((rb.r, rb.c))
    dist[rb.r][rb.c] = 0
    
    while q:
        r, c = q.popleft()

        if best_D !=-1 and best_D < dist[r][c]:
            break 

        for dx, dy in dxys[:-1]:
            nr = r + dx 
            nc = c + dy 
            
            if in_range(nr, nc) and dist[nr][nc] == -1 and \
            A[nr][nc] >= 0 and B[nr][nc] == -1:
                dist[nr][nc] = dist[r][c] + 1 
                q.append((nr, nc))

                if A[nr][nc] > 0 and (best_D == -1 or best_D == dist[nr][nc]):
                    best_D = dist[nr][nc]
                    nearest = (nr, nc) if nearest is None else min((nr, nc), nearest)

    if nearest is not None:
        B[rb.r][rb.c] = -1 
        rb.r, rb.c = nearest 
        B[rb.r][rb.c] = rb.num


# def clean(rb:Robot) -> None :
#     max_sum = 0
#     best_no_dxy = None 

#     # 하나의 로봇당 4방향에 대해 
#     for no_dxy in dxys[:-1]:  # 4방 중 제외할 방향 하나를 고른다
#         s = 0
#         for dx, dy in dxys:   # 자기 칸(0,0) + 4방
#             r, c = rb.r + dx, rb.c + dy
#             if in_range(r, c) and (dx, dy) != no_dxy:
#                 s += min(20, max(0, A[r][c])) # A[r][c]가 -1인 경우 대비하여 max(0, A[r][c])

#         if max_sum < s:
#             max_sum = s 
#             best_no_dxy = no_dxy 
    
#     if max_sum > 0:
#         for dx, dy in dxys:
#             r, c = rb.r + dx, rb.c + dy
#             if in_range(r, c) and (dx, dy) != best_no_dxy and A[r][c] > 0:
#                 A[r][c] = max(0, A[r][c] - 20)

def clean(rb:Robot) -> None:
    removed_dir = {
        0: (0, -1),
        1: (-1, 0),
        2: (0, 1),
        3: (1, 0)
    }


    cells_dict = {
        0: [(-1, 0), (0, 0), (1, 0), (0, 1)],
        1: [(0, -1), (0, 0), (0, 1), (1, 0)],
        2: [(-1, 0), (0, -1), (0, 0), (1,0)],
        3: [(-1, 0), (0, -1), (0, 0), (0, 1)]
    }

    DR = [-1, 1, 0, 0]
    DC = [0, 0, -1, 1]

    # 5가지 방향에 대한 총합 
    five_sums = A[rb.r][rb.c]
    for t in range(4):
        ar = rb.r + DR[t]
        ac = rb.c + DC[t]
        if in_range(ar, ac) and A[ar][ac] >= 0:# NOTE: -1이면 five_sum이 잘못 구해짐 
            five_sums += A[ar][ac]
    
    max_sum = 0
    max_dir = None 
    for key, dir in removed_dir.items():
        cur_sum = five_sums
        rr = rb.r + dir[0]
        rc = rb.c + dir[1]
        if in_range(rr, rc) and A[rr][rc] >= 0: # NOTE: -1이면 cur_sum이 잘못 빼짐 
            cur_sum -= min(A[rr][rc], 20)

        if cur_sum > max_sum:
            max_sum = cur_sum 
            max_dir = key 
    
    if max_sum > 0:
        for dir in cells_dict[max_dir]:
            nr = rb.r + dir[0]
            nc = rb.c + dir[1]
            if in_range(nr, nc) and A[nr][nc] > 0:
                A[nr][nc] -= min(20, A[nr][nc])

def increase() -> None:
    """먼지 축적: 모든 먼지 칸(+5). 물건(-1)이나 0칸은 변화 없음."""
    for i in range(N):
        for j in range(N):
            if A[i][j] > 0:
                A[i][j] += 5


def spread() -> None:
    """
    먼지 확산:
      - 깨끗한 칸(=0)으로만 확산됨.
      - 각 깨끗한 칸은 인접 4방의 '먼지 양 합' // 10 만큼 증가.
      - 동시 적용을 위해 temp 누적 후 일괄 반영.
    """
    temp = [[0] * N for _ in range(N)]

    for i in range(N):
        for j in range(N):
            if A[i][j] == 0:
                s = 0
                for dx, dy in dxys[:-1]:
                    r, c = i + dx, j + dy
                    if in_range(r, c) and A[r][c] > 0:
                        s += A[r][c]
                temp[i][j] = s // 10

    for i in range(N):
        for j in range(N):
            A[i][j] += temp[i][j]



def total_dust() -> int:
    """현재 격자 내 총 먼지량 합산(>0인 칸만)."""
    s = 0
    for row in A:
        for v in row:
            if v > 0:
                s += v
    return s


# L 라운드 시뮬레이션
for _ in range(L):
    # 1) 이동: 로봇 번호 순서대로
    for rb in robots:
        move(rb)

    # 2) 청소: 로봇 번호 순서대로
    for rb in robots:
        clean(rb)

    # 3) 먼지 축적
    increase()

    # 4) 먼지 확산
    spread()

    # 5) 총 먼지 출력 + 조기 종료
    s = total_dust()
    print(s)
    if s == 0:
        break