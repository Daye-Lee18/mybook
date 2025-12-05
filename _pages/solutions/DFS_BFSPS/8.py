import sys 
from collections import deque 
from heapq import heappush, heappop 
import time 

input = sys.stdin.readline 
# sys.stdin = open("Input.txt")

N, M, K = map(int, input().split())
MAX = int(1e9)
graph = []
for _ in range(N):
    graph.append(list(map(int, input().split())))

'''
필요한 데이터 구조 
'''
max_heap = []# 피공격자선정, lazy deletion 필요 
min_heap = [] # 공격자선정, lazy deletion 필요, # 공격력이 낮고, 가장 최근에 공격(max), y+x(max), x(max), potop_id 
# is_potop_alive = [True]*N*M # lazy deletion때 현재 살아있는지아닌지 확인 , graph에 그대로 기록됨. 
# potop_p_list = [0] * N*M  # 이거 다 그래프에 있는 것들임, 데이터 중복  
potop_recent_attack_time = [0] * N*M 
distance_graph = [[MAX]*M for _ in range(N)] # BFS 경로를 계속 돌 때마다 만들면, 메모리 사용량이 증가해서 reinit시켜서 사용 
# 레이저 최단 경로 우선순위: 우/하/좌/상 인데,그다음 대각선 4개 
DY = [0, 1, 0, -1, -1, -1, 1, 1]
DX = [1, 0, -1, 0, 1, -1, 1, -1]

def reinit_distance_graph():
    global distance_graph 

    for y in range(N):
        for x in range(M):
            distance_graph[y][x] = MAX 

def in_range(y, x):
    return 0 <= y < N and 0 <= x < M 

# 해당 NxM는 구라고 생각하면 쉽다. 
# 즉, 모든 격자에서 4방향/8방향 다 이어져 있다고 생각할 수 있다. 
def convert_cell_into_2Dgraph(cury, curx):
    '''
    모든 격자 위치를 받아서, 현재 cell이 graph안이면 그대로 반환, 
    아니면, 해당 cell을 graph안에 해당하는 cell로 변환해서 반환 
    '''
    if in_range(cury, curx):
        return cury, curx 
    # 만약 현재 위치가 범위 밖인 경우 범위 안으로 만들어줘야함. 
    # 8가지 범위에 대해서 먼저 대각선 cell이 범위가 더 특정되므로 먼저 해줌. 

    ##### 대각선 cell에 대해서도 
    if cury == -1 and curx == -1: # 좌측상단 
        return N-1, M-1 
    if cury == -1 and curx == M: # 우측상단 
        return N-1, 0
    if cury == N and curx == -1: # 좌측하단 
        return 0, M-1 
    if cury == N and curx == M: # 우측하단 
        return 0, 0
    
    # 범위 밖 맨 위 -> graph 안 맨 아래 
    if cury == -1: 
        return N-1, curx 

    # 범위 밖 맨 아래 -> graph 안 맨 위 
    if cury == N:
        return 0, curx 
    
    # 범위밖 맨 왼쪽 -> graph 안 오른쪽 
    if curx == -1:
        return cury, M-1 
    # 범위밖 맨 오른쪽 -> graph 안 맨 왼쪽 
    if curx == M:
        return cury, 0 
    
def make_direction(x,y):
    global N,M
    if(x < 0):
        x = N-1
    if(x >= N):
        x = 0
    if (y < 0):
        y = M-1
    if (y >= M):
        y = 0
    return x,y

def convert_locs_to_id(y, x):
    global M
    return y*M + x 

def convert_id_to_locs(id):
    y = id // M 
    x = id - y * M 
    return (y, x)

def init():
    global max_heap, min_heap
    max_heap = []
    min_heap = []
    cnt = 0 
    for y in range(N):
        for x in range(M):
            potop_id = convert_locs_to_id(y, x)
            # if graph[y][x] == 0:
            #     is_potop_alive[potop_id] = False 
            # else:
            if graph[y][x] != 0:
                cnt += 1 
                # potop_p_list[potop_id] = graph[y][x]
                heappush(max_heap, (-graph[y][x], potop_recent_attack_time[potop_id], y+x, x, potop_id)) # 정확히 반대 
                heappush(min_heap, (graph[y][x], potop_recent_attack_time[potop_id]*-1, (y+x)*-1, x*-1, potop_id)) # 공격력이 낮고, 가장 최근에 공격(max), y+x(max), x(max), potop_id 
    return True if cnt > 1 else False  

def determine_attacker():
    '''
    현재 공격자를 제외하고 남아있는 포탑의 개수가 없으면, 
    당장 중지 
    (has_to_be_stopped, attacker_id)
    '''
    while min_heap:
        power, recent_attack_time, _, _, attacker_id = heappop(min_heap)
        recent_attack_time *= -1 
        # lazy deletion 
        # if is_potop_alive[attacker_id]:
        cury, curx = convert_id_to_locs(attacker_id)
        if graph[cury][curx] > 0 : # 아직 살아있음. 
            if len(min_heap) == 0:# 남아있는 포탑의 개수가 1이 되면 바로 중지함. 
                return True, attacker_id 
            
            if graph[cury][curx] != power:
                continue # 실제 공격력과 heap에 있는 공격력이 다르면 제거 
            if potop_recent_attack_time[attacker_id] != recent_attack_time:
                continue # 실제 가장 최신 공격 시간과 다르면 제거 

            return False, attacker_id 
    

def determine_vitim(cannot_be_vitim_id):

    while max_heap:
        power, recent_attack_time, _, _, victim_id = heappop(max_heap)
        power *= -1 
        # lazy deletion 
        cury, curx = convert_id_to_locs(victim_id)
        if graph[cury][curx] == 0: # 죽은 친구는 안됨. 
            continue 
        if victim_id == cannot_be_vitim_id:
            # 자신과 같아지는 경우는 1명만 남았을 때임 
            return True, cannot_be_vitim_id
        # if potop_p_list[victim_id] != power :
        if graph[cury][curx] != power :
            continue # 공격력이 실제와 다르면 제거 
        if potop_recent_attack_time[victim_id] != recent_attack_time:
            continue # 공격시간도 실제로 다르면 제거 
        return False, victim_id 

# print(convert_locs_to_id(0,0)) # 0
# print(convert_locs_to_id(1,3)) # 7
# print(convert_locs_to_id(3,3)) # 15 
# print(convert_id_to_locs(7)) # (1, 3)
# print(convert_id_to_locs(14)) # (3, 2)
# print(convert_id_to_locs(10)) # (2, 2)

def can_attack_by_lazer(attacker_id, victim_id):
    '''
    레이저는 상하좌우 방향으로 움직임 가능. 
    부서진 포탑 즉, 그래프에서 0이된 지점은 움직일 수 없음. 
    '''
    q = deque()
    s_y, s_x = convert_id_to_locs(attacker_id)
    e_y, e_x = convert_id_to_locs(victim_id)
    q.append((e_y, e_x))
    reinit_distance_graph() # XXX: 반드시 먼저 해줘야함. 
    distance_graph[e_y][e_x] = 0
    
    while q:
        cury, curx = q.popleft()

        if cury == s_y and curx == s_x:
            break 
            
        
        # 우/하/좌/상의 우선순위대로 먼저 움직인 경로 선택 
        # XXX: 도착점에서 -> 출발점까지의 거리를 나타낸 것이므로, 
        # 상/좌/하/우순으로 바라봐야함. 거꾸로!  
        for t in range(4):
            nxt_y = cury + DY[t]
            nxt_x = curx + DX[t]

            # 해당 맵은 "구"이기 때문에 graph밖으로 벗어날 수 없고 무조건 통과됨. 
            nxt_y, nxt_x = convert_cell_into_2Dgraph(nxt_y, nxt_x)

            if graph[nxt_y][nxt_x] == 0: # 부서진 포탑의 위치로는 갈 수 없음 
                continue 
            
            # 이미 지나온 경로인지, 파악하려면, distance list를 따로 만들어야하지만, 
            # 어차피 이때까지의 경로도 같이 출력해야함. 
            if distance_graph[nxt_y][nxt_x] == MAX: # 한 번 지나간 경로가 아님. 
                q.append((nxt_y, nxt_x))
                distance_graph[nxt_y][nxt_x] = distance_graph[cury][curx] + 1 
    
    result = set()
    if distance_graph[s_y][s_x] == MAX: # 도달 불가능 
        return False, result # 레이저 공격 불가능, 아무것도 업는 deque 반환 
    
    # 시작점에서 끝 점까지 path 경로를 알아야함. 
    # 이때 경로의 순서를 알필요는 없음. 
    # XXX: 공격자와 피공격자를 제외한 경로에 있는 애들만 알면 됨.
    q = deque() 
    q.append((s_y, s_x))

    while q:
        cury, curx = q.popleft()

        if cury == e_y and curx == e_x:
            # 마지막 victim_id는 result에 넣지 않음. 
            break 

        # 우/하/좌/상의 우선순위대로 먼저 움직인 경로 선택 
        for t in range(4):
            nxt_y = cury + DY[t]
            nxt_x = curx + DX[t]

            # 해당 맵은 "구"이기 때문에 graph밖으로 벗어날 수 없고 무조건 통과됨. 
            nxt_y, nxt_x = convert_cell_into_2Dgraph(nxt_y, nxt_x)

            # 0인 위치는 이미 distance가 MAX여서 아래 조건만 만족시키면 됨. 
            # if graph[nxt_y][nxt_x] == 0: # 부서진 포탑의 위치로는 갈 수 없음 
            #     continue 
            # cost가 작아지는 경로로 이동 
            if distance_graph[nxt_y][nxt_x] < distance_graph[cury][curx]: # 한 번 지나간 경로가 아님. 
                q.append((nxt_y, nxt_x))
                result.add((nxt_y, nxt_x))
                break # 한 번 넣으면 다음 경로 
    return True, result # 레이저 공격 가능, 최소 경로 반환  

def debug():
    print(f"potop 공격력: ", graph)
    print(f"min_heap: ", min_heap)
    print(f"max_heap: ", max_heap)

if __name__ == "__main__":
    # start_time = time.time()
    init() # 포탑과 관련한 필요한 데이터 정리 
    # debug()
    for cur_time in range(1, K+1): # O(1000)
        # step 1: 공격자 선정 
        has_to_be_stopped, current_attacker_id = determine_attacker() # O(log(N*M))

        if has_to_be_stopped:
            break 
        
        attacker_y, attacker_x = convert_id_to_locs(current_attacker_id)
        # 공격자가 선정된 후에는, max_heap과 min_heap은 (남아있는 공격력, 공격 시간)에 의존하므로, update 필요하게 되는데, 
        # 일단 공격자의 공격시간이 정해졌으므로 업데이트 
        graph[attacker_y][attacker_x] += (N+M)
        potop_recent_attack_time[current_attacker_id] = cur_time 
        

        # action 2: 피공격자 선정: 공격자를 제외한 가장 강한 포탑을 공격해야함. 
        has_to_be_stopped, victim_id = determine_vitim(current_attacker_id) # O(log(N*M))
        victim_y, victim_x = convert_id_to_locs(victim_id)

        if has_to_be_stopped:
            break 

        # action 2-1. 레이저 공격이 가능한지 확인 
        yes_you_can, lazer_path = can_attack_by_lazer(current_attacker_id, victim_id)


        if yes_you_can:
            # action2-1.: 레이저 공격 실시 

            # 공격대상은 공격력만큼 피해를 받음, 최소 0이여야함. 
            graph[victim_y][victim_x] = max(0, graph[victim_y][victim_x]-graph[attacker_y][attacker_x])
            # 공격대상 제외 공격경로에 있는 피해자들 
            for vy, vx in lazer_path:
                # vy, vx= convert_id_to_locs(in_a_way_victim_id)
                if vy == victim_y and vx == victim_x:
                    continue 
                graph[vy][vx] = max(0, (graph[vy][vx] - (graph[attacker_y][attacker_x] // 2)))
        else: 
            # action 2-2. 포탄 공격 
            # graph안의 값과 power_list모두 업데이트 해줘야함. 
            graph[victim_y][victim_x] = max(0, graph[victim_y][victim_x]-graph[attacker_y][attacker_x])
            # 공격"비대상자"의 8개 주의 위치에 대해서 절반 만큼의 피해를 받는다. 
            # 하지만 "공격자 자신"은 포탄의 피해를 받지 않는다. 
            # 아래의 정보도 계속 가지고 있어야함. 
            eight_cells_except_attacker = set() 
            for t in range(8):
                nxty = victim_y + DY[t]
                nxtx = victim_x + DX[t]

                nxty, nxtx = convert_cell_into_2Dgraph(nxty, nxtx)
                if nxty == attacker_y and nxtx == attacker_x:
                    continue 
                graph[nxty][nxtx] = max(0,  graph[nxty][nxtx]-(graph[attacker_y][attacker_x] // 2))
                eight_cells_except_attacker.add((nxty, nxtx))


            
        # action 4: 포탑 정비. 포탑과 무관한 포탑의 공격력은 1을 받음 
        for y in range(N):
            for x in range(M):
                if graph[y][x] == 0:
                    continue 
                if y == attacker_y and x == attacker_x:
                    continue 
                if y == victim_y and x == victim_x:
                    continue  
                # 레이저 공격을 했으면, lazer경로에 있으면 안됨. 
                if yes_you_can and (y, x) in lazer_path:
                    continue 
                # 포탄 공격에는 그 영향력 안에 있으면 안됨. 
                elif not yes_you_can and (y, x) in eight_cells_except_attacker:
                    continue 
                # 위의 모든 것이 아닐 때 
                graph[y][x] += 1 
        
        # step 4: 
        # 다음 턴으로 들어가기 전에, max_heap, min_heap의 정보를 다시 정리하는 것이 필요함. 
        # 즉, 공격자와 이번 턴에서 죽은 친구들을 제외하고, 공격받은 친구들은 다 graph안의 공격력이 변해있음. 
        # 또, 공격자도 최신 공격시간이 변했기 때문에, 데이터가 변함. 
        can_go_nxt_turn = init() # max_heap과 min_heap에 대해 데이터를 새로이 넣어줌. 
        
        if not can_go_nxt_turn:
            break # turn이 더이상 종료되지 못함. 남아있는 포탑이 1개 이하임. 

    
    # 최종적으로 가장 강한 포탑의 공격력 (max_heap)의 맨 처음 원소 
    has_to_be_stopped, last_victim_id = determine_vitim(-1) # 공격자의 아이디는 -1로 아예 존재하지 않는 것 추가 
    final_y, final_x = convert_id_to_locs(last_victim_id)
    print(graph[final_y][final_x])

    # end_time = time.time()
    # print(f"time collapse: {(end_time - start_time)*1e3}")