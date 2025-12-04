import sys 
from collections import deque 

input = sys.stdin.readline 
# sys.stdin = open("Input.txt")
N, M, F = map(int, input().split())

'''
필요한 데이터 구조
'''
MAX = int(1e9)
graph = []
# 시간의 벽면도 
wall_graph = [[5]*3*M for _ in range(3*M)] # 5 (없는 공간) 장애물로 초기화 
distance_wall_graph = [[MAX]*3*M for _ in range(3*M)]
distance_graph = [[MAX]*N for _ in range(N)]
weird_time_graph = [[-1]*N for _ in range(N)]
weird_time_wall_graph = [[-1]*3*M for _ in range(3*M)]
f_v_list = []
f_d_list = [] 
f_y_list = []
f_x_list= [] 
is_f_id_in_wall = [False]*F 
has_diffusion_stopped = [False]*F 
howmany_time_diffused = [0]*F 
s_y, s_x = None, None # 시간의 벽 윗면에서 시작 위치 
e_y_first, e_x_first = None, None # 시간의 벽에서 도착 위치 
s_y_second, s_x_second = None, None # 단면도에서 시작 위치 
e_y_final, e_x_final = None, None # 단면도 최종 도착 위치 
DY = [-1, 1, 0, 0]; DX = [0, 0, -1, 1]
# 첫번째 도착지점이 여기에 있음 
flag = 0
flag2 = 0 
for _ in range(N):
    row = list(map(int, input().split()))
    graph.append(row[:])
    
    # 두번째 시작 지점 찾기 
    if flag == 0:
        for idx in range(N):
            if row[idx] == 3:
                upper_left_most_y = len(graph) -1 
                upper_left_most_x = idx 
                flag = 1  # 한번찾으면 안찾아도 됨. 
                break 

    # 두번째 도착지점 찾기 
    if flag2 == 0:
        for idx in range(N):
            if row[idx] == 4:
                e_y_final = len(graph) -1 
                e_x_final = idx 
                flag2 = 1  # 한번찾으면 안찾아도 됨. 
                break 


# 처음 3을 찾고, MxM에서 옆에 0이 있는 그지점이 단면도까지의 도착 위치 
flag = 0
# print(f"upper y, x{upper_left_most_y, upper_left_most_x}")
for y in range(upper_left_most_y-1, upper_left_most_y+M+1):
    for x in range(upper_left_most_x-1, upper_left_most_x+M+1):
        
        if graph[y][x] == 0:
            s_y_second = y  
            s_x_second = x 
            flag = 1 
            break 
    if flag:
        break 

# 단면도상 (N) 에서의 첫번째 도착위치 -> 시간의 벽 상(3*M) 에서의 도착 위치 
def transform_second_start_to_first_end(cur_y, cur_x):
    # global s_y_second, cur_x, e_y_first, e_x_first 

    # # 단면도에서 의 최외곽 
    # outer_list = []
    # for i in range(M): # 맨 윗줄 
    #     outer_list.append((upper_left_most_y-1, upper_left_most_x+i))
    # for i in range(M): # 맨 아랫줄 
    #     outer_list.append((upper_left_most_y+M, upper_left_most_x+i))
    # for i in range(M): # 맨 왼쪽 줄 
    #     outer_list.append((upper_left_most_y+i, upper_left_most_x-1))
    # for i in range(M): # 맨 오른쪽 
    #     outer_list.append((upper_left_most_y+i, upper_left_most_x+M))

    # print(f"outer_list: {outer_list}") # 4*M 의 length 

    returny, returnx = None, None 
    # 위 
    if cur_y == upper_left_most_y-1 and upper_left_most_x <= cur_x < upper_left_most_x + M:
        returny = 0
        diff = cur_x - upper_left_most_x
        returnx = M + diff 
    # 아래 
    elif cur_y == upper_left_most_y + M and upper_left_most_x<=cur_x < upper_left_most_x + M:
        returny = 3*M -1 
        diff = cur_x - upper_left_most_x
        returnx = M + diff 
    # 왼쪽 
    elif upper_left_most_y <= cur_y < upper_left_most_y + M and cur_x == upper_left_most_x -1:
        diff = cur_y - upper_left_most_y
        returny = M + diff 
        returnx = 0 
    # 오른쪽 
    elif upper_left_most_y <= cur_y < upper_left_most_y + M and cur_x == upper_left_most_x + M:
        diff = cur_y - upper_left_most_y
        returny = M + diff 
        returnx = 3*M -1 

    return returny, returnx
 

    
e_y_first, e_x_first = transform_second_start_to_first_end(s_y_second, s_x_second)


def rotation_90CCW(original_graph):
    '''
    (y, x) -> (M-x-1, y)
    '''
    new_graph = [[0]*M for _ in range(M)]
    for y in range(M):
        for x in range(M):
            new_graph[M-x-1][y] = original_graph[y][x]
    
    for idx in range(M):
        original_graph[idx] = new_graph[idx][:] # update 

new_graph = [[0]*M for _ in range(M)]
for d in range(5): # 동서남북윗면 평면도 순 
    if d == 0: # 동쪽이면, 왼쪽으로 90도 회전 
        
        for i in range(M):
            # wall_graph[M+i][2*M:] = list(map(int, input().split()))
            new_graph[i] = list(map(int, input().split()))
        rotation_90CCW(new_graph)

        for i in range(M):
            wall_graph[M+i][2*M:] = new_graph[i]

    elif d == 1: # 서쪽이면, 왼쪽으로 270도 회전 
        for i in range(M):
            # wall_graph[M+i][:M] = list(map(int, input().split()))
            new_graph[i] = list(map(int, input().split()))
        rotation_90CCW(new_graph)
        rotation_90CCW(new_graph)
        rotation_90CCW(new_graph)

        for i in range(M):
            wall_graph[M+i][:M] = new_graph[i]

    elif d == 2:  # 남쪽 그대로 
        for i in range(M):
            wall_graph[2*M+i][M:2*M] = list(map(int, input().split()))

    elif d == 3: # 북쪽이면, 왼쪽으로 180도 회전 
        for i in range(M):
            # wall_graph[i][M:2*M] = list(map(int, input().split()))
            new_graph[i] = list(map(int, input().split()))

        rotation_90CCW(new_graph)
        rotation_90CCW(new_graph)

        for i in range(M):
            wall_graph[i][M:2*M] = new_graph[i]


    elif d == 4: # 윗면 
        # 이 윗면에 시작점이 존재함. 
        for i in range(M):
            wall_graph[M+i][M:2*M] = list(map(int, input().split()))
        
            for x in range(M, 2*M):
                if wall_graph[M+i][x] == 2:
                    s_y = M+i 
                    s_x = x 


# 각 Cell마다 dy, dx 설정
DY = [0, 0, 1, -1] # 동서남북  0/1/2/3
DX = [1, -1, 0, 0]

def find_next_cell(y, x):
    '''
    특정 셀이 아니면 (-1, -1)를 반환 
    '''
    # 좌측상단 설정 
    if 0 <= y < M and x == M:
        nxty = x
        nxtx = y
    elif 0<= x < M and y == M:
        nxty = x
        nxtx = y
    # 우측상단설정 
    elif 0 <= y < M and x == 2*M -1 :
        nxty = 3*M-x-1 
        nxtx = 3*M-y-1
    elif y == M and 2*M <= x < 3*M:
        nxty = 3*M-x-1
        nxtx = 3*M-y-1
    # 우측하단 설정 
    elif y == 2*M -1 and 2*M <= x < 3*M:
        nxty = x 
        nxtx = y 
    elif 2*M <= y <= 3*M and x == 2*M -1 :
        nxty = x 
        nxtx = y 
    # 좌측하단 설정 
    elif y == 2*M -1 and 0 <= x < M:
        nxty = 3*M - x -1 
        nxtx = 3*M - y -1 
    elif 2*M <= y < 3*M and  x == M:
        nxty = 3*M - x -1
        nxtx = 3*M - y -1 
    else: # 그외의 공간은 방향이 없음 
        nxty = -1 
        nxtx = -1 
    return nxty, nxtx 


# print(find_next_cell(0, 3)) # (3,0)
# print(find_next_cell(1, 3)) # (3,1)
# print(find_next_cell(0, 5)) # (3, 8)
# print(find_next_cell(1, 5)) # (3, 7)
# print(find_next_cell(3, 0)) # (0, 3)
# print(find_next_cell(5, 6)) # (6, 5)
# print(find_next_cell(5, 8)) # (8, 5)
# print(find_next_cell(5, 0)) # (8, 3)
# print(find_next_cell(5, 1)) # (7, 3)
# print(find_next_cell(0, 0)) # (-1, -1)
# print(find_next_cell(3, 3)) # (-1, -1)

def wall_in_range(y, x):
    global M
    return 0<=y<3*M and 0<=x<3*M

def graph_in_range(y, x):
    global N 
    return 0<=y<N and 0<=x < N 

for f_id in range(F):  
    y, x, time_d, v = map(int, input().split())
    # 미지의 바닥 공간에만 존재. 
    weird_time_graph[y][x] = f_id 
    f_v_list.append(v) 
    f_d_list.append(time_d)
    f_y_list.append(y)
    f_x_list.append(x)

# 단면도 -> wall안에서 확산하는 과정을 list에 저장  
# idx로 돌면서, 해당 시간이 지나면, 그 idx에 있도록 하기 
def diffusion_f(time):
    global F, f_v_list

    if time == 0: # 확산 필요 없음 
        return 

    for f_id in range(F):
        if has_diffusion_stopped[f_id]:
            continue 

        # XXX: 0으로 나눌때마다 diffusion하면 안됨 
        # BFS에서는 같은 시간 curtime 이 7번이 여러번 나올수도 있음. 
        # 현재 시간 5//5 == 1 이면, 기존 diffusion 횟수가 0번이여야 diffusion 가능 
        if time % f_v_list[f_id] == 0 and time // f_v_list[f_id] == howmany_time_diffused[f_id] + 1 :
            if not is_f_id_in_wall[f_id]: # 단면도에서 확산 
                # 확산해야함. 확산 방향 1개 
                f_dir = f_d_list[f_id] # 0/1/2/3 동서남북 
                f_y = f_y_list[f_id]
                f_x = f_x_list[f_id]
                nxty = f_y + DY[f_dir] ; nxtx = f_x + DX[f_dir]
                if not graph_in_range(nxty, nxtx): # 범위밖 
                    has_diffusion_stopped[f_id] = True 
                    continue 
                if graph[nxty][nxtx] == 1 and graph[nxty][nxtx] == 4:
                    has_diffusion_stopped[f_id] = True 
                    continue # 장애물과 탈출구가 있으면 움직일 수 없음. 
                    
                # 확산 가능 
                if graph[nxty][nxtx] == 3: # 다음 방향이 벽면임 
                    # ㅋㅋㅋㅋㅋ 
                    is_f_id_in_wall[f_id] = True 
                    # 현재 단면도 -> 시간의 벽 상의 포지션으로 바꿈 
                    wally, wallx = transform_second_start_to_first_end(nxty, nxtx)

                    if wall_graph[wally][wallx] == 1: # 그래프상 1이면 
                        has_diffusion_stopped[f_id] = True 
                        continue # 확산 stop 
                    else: # 그래프에 0이나 2면, 확산 가능 
                        weird_time_wall_graph[wally][wallx] = f_id 
                        f_y_list[f_id] = wally 
                        f_x_list[f_id] = wallx 
                        howmany_time_diffused[f_id] += 1 

                elif graph[nxty][nxtx] == 0:
                    weird_time_graph[nxty][nxtx] = f_id # 확산 
                    f_y_list[f_id] = nxty 
                    f_x_list[f_id] = nxtx 
                    howmany_time_diffused[f_id] += 1 

            else: # 시간의 벽에서 확산 
                # 확산해야함. 확산 방향 1개 
                f_dir = f_d_list[f_id] # 0/1/2/3 동서남북 
                f_y = f_y_list[f_id]
                f_x = f_x_list[f_id]
                nxty = f_y + DY[f_dir] ; nxtx = f_x + DX[f_dir]
                if not wall_in_range(nxty, nxtx): # 범위밖 
                    has_diffusion_stopped[f_id] = True 
                    continue 
                if wall_graph[nxty][nxtx] == 1:
                    has_diffusion_stopped[f_id] = True 
                    continue # 장애물이있으면 움직일 수 없음. 
                else:
                    weird_time_wall_graph[nxty][nxtx]= f_id 
                    f_y_list[f_id] = nxty 
                    f_x_list[f_id] = nxtx 
                    howmany_time_diffused[f_id] += 1 

            

def debug():
    for row in wall_graph:
        print(row)

    print(f"Start point: {s_y, s_x}")
    print(f"First End point on time wall: {e_y_first, e_x_first}")
    print(f"Second Start point: {s_y_second, s_x_second}")
    print(f"Second End point: {e_y_final, e_x_final}")


    print(f"first total time: {first_total_time}")
    print(f"total_time : {total_time}")



if __name__ == "__main__":
    # Step1: 시간의 벽에서 미지의 공간의 바닥의 출구로 경로 탐색 
    total_time = 0
    q = deque()
    q.append((s_y, s_x, 0)) # 타임머신 위치, 현재 시간 
    distance_wall_graph[s_y][s_x] = 0 # 3*M x 3xM에서 여행 
    
    find_first_path = 0
    while q:
        cury, curx, curtime = q.popleft()

        if cury == e_y_first and curx == e_x_first:
            find_first_path = 1 
            total_time = curtime 
            break 

        # sumF = sum(1 if has_diffusion_stopped[f_id] else 0 for f_id in range(F))
        # if sumF == F and 
        # 시간이상현상 확산 
        # 만약 현재 시간이 v_i의 배수이면, 타임머신이 이동해있음 
        diffusion_f(curtime)

        # 만약 현재 시간에 시간의 벽에 출구를 통해 시간 이상 현상이 확산되면, 길이 없음 
        if weird_time_wall_graph[e_y_first][e_x_first] != -1:
            # print(-1)
            break 

        # 타임머신 이동 
        for t in range(4): # 동서남북 
            nxty = cury + DY[t]
            nxtx = curx + DX[t]

            if not wall_in_range(nxty, nxtx): # 격자 밖인 경우 
                continue 

            if wall_graph[nxty][nxtx] == 1 : # 장애물이 있는 경우 
                continue 
            
            if wall_graph[nxty][nxtx] == 5 : # 원래 범위가 아닌 경우 
                continue 

            if weird_time_wall_graph[nxty][nxtx] != -1: # 시간 이상 현상 존재 
                continue 
            # 위의 경우가 아니면 이동 가능 
            if distance_wall_graph[nxty][nxtx] == MAX:
                q.append((nxty, nxtx, curtime +1))
                distance_wall_graph[nxty][nxtx] = curtime + 1 
                # wall_graph[nxty][nxtx] = 2 ### DEBUG 

        nxty, nxtx = find_next_cell(cury, curx)
        if nxty == -1 and nxtx == -1:
            continue # 벽면을 타고 움직일 수 없음 
        else:
            if wall_graph[nxty][nxtx] == 1:
                continue # 장애물이 있음 
            if weird_time_wall_graph[nxty][nxtx] != -1: # 시간이상현상 존재 
                continue 
            # 위의 경우가 아니면 이동 가능 
            if distance_wall_graph[nxty][nxtx] == MAX:
                q.append((nxty, nxtx, curtime +1 ))
                distance_wall_graph[nxty][nxtx] = curtime + 1 
                # weird_time_wall_graph[nxty][nxtx] = 2 
                # wall_graph[nxty][nxtx]= 2 

    if find_first_path == 0:
        print(-1)
    else: # 시간의 벽에서 탈출 성공 
        # print(total_time)
        first_total_time = total_time 
        # 첫번째 도착지점에서 -> 두번째 도착지점까지 1시간 걸림 
        total_time += 1 
        # 바닥에서 탈출하기 

        q = deque()
        q.append((s_y_second, s_x_second, total_time))
        distance_graph[s_y_second][s_x_second] = 0 
        find_second_path = 0 

        while q:
            cury, curx, curtime = q.popleft()

            # 현재 시간, 현재 위치에 이상 현상 있으면 제거 
            if weird_time_graph[cury][curx] != -1:
                continue 
            # if cury == e_y_final and curx == e_x_final:
            if graph[cury][curx] == 4: # 도착지점 
                print(curtime)
                total_time = curtime 
                find_second_path = 1 
                break 
            
            diffusion_f(curtime)

            # 타임머신 이동 
            for t in range(4): # 동서남북 
                nxty = cury + DY[t]
                nxtx = curx + DX[t]

                if not graph_in_range(nxty, nxtx): # 격자 밖인 경우 
                    continue 

                if graph[nxty][nxtx] == 1: # 장애물이 있는 경우 
                    continue 

                if graph[nxty][nxtx] == 3: # 다시 시간의 벽으로 들어감 
                    continue 

                if weird_time_graph[nxty][nxtx] != -1: # 시간 이상 현상 존재 
                    continue 

                # 위의 경우가 아니면 이동 가능 
                if distance_graph[nxty][nxtx] == MAX:
                    q.append((nxty, nxtx, curtime +1))
                    distance_graph[nxty][nxtx] = curtime + 1 
                    # wall_graph[nxty][nxtx] = 2 ### DEBUG 

        if find_first_path and find_second_path == 0:
            print(-1)
                
    # debug()   