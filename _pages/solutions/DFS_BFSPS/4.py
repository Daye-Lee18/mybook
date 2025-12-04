from collections import deque, defaultdict 
import sys 

input = sys.stdin.readline 
# sys.stdin = open("Input.txt")

# 마을의 크기 N, 전사의 수 M
N, M = map(int, input().split())
# 메두사의 집 위치, 공원 위치 
s_y, s_x, e_y, e_x = map(int, input().split())

# 상하좌우 우선순위 
DY = [-1, 1, 0, 0]; DX = [0, 0, -1, 1]
'''
필요한 자료구조
'''

warriors_graph = defaultdict(list) # dict((y, x) -> list of idx) 현재 전사들의 위치 관리, (y,x)위치에 -> idx 보관
MAX = int(1e9)
distances_graph = [[MAX] *N for _ in range(N)]
war_ys = [0] * M 
war_xs = [0] * M
alive = [True] * M 
visited = [[] for _ in range(M)] # 전사들이 이동할 때, 메두사의 매 turn마다 set()을 초기화하는 것은 메모리 사용량이 많아지므로, 계속 초기화 하면서 관리한다. 

class Medusa:
    def __init__(self, y:int, x:int):
        self.y = y
        self.x = x 
    def set_loc(self, y, x):
        self.y = y 
        self.x = x 
# --------------

medusa_object = Medusa(s_y, s_x)

# 전사의 시작 위치 
warriors_locs = list(map(int, input().split()))
cnt = 0
for i in range(0, len(warriors_locs), 2):
    y, x = warriors_locs[i], warriors_locs[i+1]
    warriors_graph[(y, x)].append(cnt)
    war_ys[cnt] = y 
    war_xs[cnt] = x 
    cnt += 1 
    

# 마을 도로 정보 
graph = []
for _ in range(N):
    graph.append(list(map(int, input().split()))) # 도로 0, 비도로 1 

def in_range(y, x):
    global N 
    return 0<=y < N and 0 <= x < N

def build_distances():# distances_graph를 채움 
    q = deque()
    q.append((e_y, e_x))
    distances_graph[e_y][e_x] = 0

    while q:
        y, x= q.popleft()
        for t in range(4):
            ny = y + DY[t]; nx = x + DX[t]

            if in_range(ny,nx) and distances_graph[ny][nx] == MAX and graph[ny][nx] == 0:
                distances_graph[ny][nx] = distances_graph[y][x] + 1 
                q.append((ny, nx))


# 메두사의 집에서 공원까지 최단 경로 파악, weight이 1인 graph에서는 BFS사용 가능 
# 즉, dijkstra algorithm에서 graph의 edge weight=1인 특수 케이스가 BFS   
def dijkstra():
    global s_y, s_x , e_y, e_x, distances_graph
    q = deque()
    q.append((s_y, s_x))
    path = deque()
    path.append((s_y, s_x))
    flag = 0 

    while q:
        cury, curx = q.popleft()
        # path.append((cury, curx))
        
        if cury == e_y and curx == e_x:
            flag = 1 
            break 
        
        cur_dis = distances_graph[cury][curx]
        for t in range(4):
            ny = cury + DY[t]; nx = curx + DX[t]
            if in_range(ny, nx) and graph[ny][nx] == 0 and \
                cur_dis > distances_graph[ny][nx]: # 도로로만 이동가능, distances_graph는 0/1 도로/비도로 상황을 고려하여 도착점까지의 거리를 계산한 그래프로, 해당 길이 있다면, 항상 더 거리가 짧은 다음 노드가 존재하게 됨. 
                q.append((ny, nx))
                path.append((ny,nx))
                break # 하나의 방향을 상/하/좌/우로 찾았으면 바로 다음 길 찾으면 됨. 
                
                    
    return path if flag else deque()

def calculate_vision(medusa_y, medusa_x):
    # direction : 상하좌우 
    max_cnt = -1
    return_vision_map = None
    best_dir = None 

    VISION_DXYS = [
        [(-1, -1), (-1, 0), (-1, 1)], # 상 t= 0, dy = -1
        [(1, -1), (1, 0), (1, 1)], # 하 t= 1, dy = 1
        [(-1, -1), (0, -1), (1, -1)], # 좌 t = 2 , dx = -1
        [(-1, 1), (0, 1), (1, 1)] # 우 t = 3 , dx = 1 
    ]
    # 4가지 방향에 대해서 
    for t in range(4):
        '''
        전사의 수는 최대 300명이 될 수 있는데, 그렇게 되면, 아래의 데이터 구조가 너무 많아서, 메모리가 터질 수 있다.
        따라서, 하나의 vision map만 만들어서, 관리한다. 
        즉, 별도의 Locs_set 두개는 사용하지 않음. 

        Algorithm 
        q는 BFS에서 현재 위치, 
        warrior_pq는 시야각에 있던 전사들의 모음 
        vision[y][x]는 메두사의 시야를 나타냄 1: 시야각에 존재. 
        '''
        # all_possible_locs_set = set()
        # all_warrior_blocks_set = set()
        q = deque()
        vision = [[0]*N for _ in range(N)]  # 해당 격자가 메두사의 시야각에 위치하면 1이됨. 이걸로 visited 도 판별 가능 
        warrior_pq = deque() 
        warrior_cnt = 0 

        # q INIT 
        for dir in range(3):
            ny = medusa_y + VISION_DXYS[t][dir][0]
            nx = medusa_x + VISION_DXYS[t][dir][1]
            
            if in_range(ny, nx) and vision[ny][nx] == 0:
                q.append((ny, nx, dir))
                vision[ny][nx] = 1 
                
                if warriors_graph[(ny, nx)]:
                    # lazy_deletion_warriors_graph(ny, nx)
                    # if warriors_graph:
                    warrior_pq.append((ny, nx, dir))

        # 모든 시야각 vision에 채움 
        while q:
            cury, curx, curdir = q.popleft()

            # 메두사로부터 내려온 방향에 그대로 사용하도록 
            # NOTE: 각 cell을 방문할 때 중복이 없도록 하는 것이 포인트! 만약에 3방향 모두 허용하면, 겹쳐서 틀리게됨. 
            if curdir == 0: # dy / dx = -1 
                start = 0; end = 2 # inclusive ~ exclusive 
            elif curdir == 1:
                start = 1; end = 2 # exclusive 
            else:
                start = 1; end = 3 # exclusive 

            for i in range(start, end): 
                nxt_y = cury + VISION_DXYS[t][i][0]
                nxt_x = curx + VISION_DXYS[t][i][1]
                # 해당 격자가 아직 시야각에 없는 경우 0의 값을 가짐 
                if in_range(nxt_y, nxt_x) and vision[nxt_y][nxt_x] == 0: 
                    q.append((nxt_y, nxt_x, curdir)) # curdir는 계속 가지고 있음. 
                    vision[nxt_y][nxt_x] = 1  # 메두사 시야각에 존재 
                    # all_possible_locs_set.add((nxt_y, nxt_x))
                if in_range(nxt_y, nxt_x) and warriors_graph[(nxt_y, nxt_x)]:
                    # lazy_deletion_warriors_graph(nxt_y, nxt_x)
                    # if warriors_graph:
                    warrior_pq.append((nxt_y, nxt_x, curdir))

        # 해당 warrior로부터 가려지는 부분 다시 지움 
        while warrior_pq:
            cury, curx, curdir = warrior_pq.popleft()

            if curdir == 0: # dy / dx = -1 
                start = 0; end = 2 # inclusive ~ exclusive 
            elif curdir == 1:
                start = 1; end = 2 # exclusive 
            else:
                start = 1; end = 3 # exclusive 
            
            for i in range(start, end): # 0, 1
                nxt_y = cury + VISION_DXYS[t][i][0]
                nxt_x = curx + VISION_DXYS[t][i][1]
                if in_range(nxt_y, nxt_x) and vision[nxt_y][nxt_x] == 1: # 현재 코드에서 전개하는 방향들에 대해서 겹치는 부분이 있을 수 없으므로 (nxt_y, nxt_x) in all_possible_locs_set 조건 체크는 불필요 
                    warrior_pq.append((nxt_y, nxt_x, curdir))
                    vision[nxt_y][nxt_x] = 0 # 다시 가리기 

        # 최종 방향 t에서 메두사 시야각은 vision에 담아있음 
        for y in range(N):
            for x in range(N):
                if vision[y][x] == 1: # 시야에 있는데 
                    if warriors_graph[(y, x)]: # 그 시야안에 전사들이 있다면, 
                        cnt = len(warriors_graph[(y, x)]) # 이미 lazy deletion됐음. 
                        warrior_cnt += cnt 
        
        if warrior_cnt > max_cnt:
            max_cnt = warrior_cnt 
            return_vision_map = vision 
            best_dir = t


    return max_cnt, return_vision_map, best_dir 
        

def calculate_mahattan_distance(y1, x1, y2, x2):
    return abs(y1- y2) + abs(x1-x2)

def reinit_visit(war_id):
    global visited 
    while visited[war_id]:
        visited[war_id].pop()

if __name__ == "__main__":
    build_distances() # 끝점 (e_y, e_x)에서 시작점 까지의 최단 거리 disatnces_graph 채움 
    path = dijkstra() # path는 (s_y, s_x)에서 끝점 (e_y, e_x)까지 최단 거리로 가는 길 반환 
    if path:
        path.popleft() # 맨 처음 위치 생략 
        # path에 들어있는 길만큼 턴이 진행됨. 
        while path:
            # 살아있는 전사들은 모두 is_stoned에서 풀려남 

            # step1. 메두사의 이동 
            nxt_y, nxt_x = path.popleft()
            if nxt_y == e_y and nxt_x == e_x:
                print(0)
                break # 메두사가 공원에 도착하면, 0을 출력 후 종료 
            
            medusa_object.set_loc(nxt_y, nxt_x)
            # 메두사가 이동한 후 위치에 전사들이 있으면 공격을 받고 사라짐. 
            warriors_ids = warriors_graph[(nxt_y, nxt_x)] 
            if warriors_ids:
                for w in warriors_ids:  
                    alive[w] = False 
                    
                # 현재 위치는 다 동일하므로, 아래에 대해서는 한 번만 없애주면 됨.
                warriors_graph[(nxt_y, nxt_x)] = [] # 빈 집합으로 전사들 모두 정리 
            
            # step2. 메두사의 시선 
            num_stoned_warriors, vision_map, best_dir = calculate_vision(medusa_object.y, medusa_object.x)
            
            # step3. 전사들의 이동 
            reached_warriors = []
            total_steps = 0
            
            # 살아있는 전사들에 대해서
            for war_id in range(M):

                if not alive[war_id]: # 죽은 전사들 pass 
                    continue 
    
                if vision_map[war_ys[war_id]][war_xs[war_id]] == 1: # 살아있지만, 메두사의 시야에 있는 전사들은 돌이됨. 
                    continue # 이번 턴은 돌이 돼서 움직일 수 없음. 
                
                move_cnt = 0
                q = deque()
                w_y = war_ys[war_id]; w_x = war_xs[war_id]
                
                reinit_visit(war_id) # visited 관리, 메모리 사용량 줄이기 위해, 시작하기 전에 빈 리스트로 만들어줌. 
            
                cur_best_distance = calculate_mahattan_distance(w_y, w_x, medusa_object.y, medusa_object.x)
                q.append((w_y, w_x, cur_best_distance))
                
                while move_cnt <= 2:
                    cur_y, cur_x, cur_dis = q.popleft()
                    # distance_between_medusa = calculate_mahattan_distance(cur_y, cur_x, medusa_object.y, medusa_object.x)
                    
                    if cur_dis == 0: # 0 <= move_cnt가 <= 2일 때 warriors가 메두사의 위치와 동일하면,
                        reached_warriors.append(war_id)
                        break 

                    if move_cnt == 2: # 
                        break 
                    # if distance_between_medusa >= cur_best_distance:  # 상하좌우로 우선순위 선택 
                    #     continue 
                    flag = 0
                    if move_cnt == 0:
                        range_list = range(4)
                    elif move_cnt == 1:
                        range_list = range(2, 6 ,1)
                    for t in range_list: # 상하좌우로 우선순위 선택 
                        # print(f"range_list: {range_list}")
                        ny = cur_y + DY[t%4] ; nx = cur_x + DX[t%4]
                        if in_range(ny, nx) and \
                        vision_map[ny][nx] == 0 and \
                            not (ny, nx) in visited[war_id]:
                            new_distance_between_medusa = calculate_mahattan_distance(ny, nx, medusa_object.y, medusa_object.x)
                            # 한번 업데이트 하면 for loop break 
                            if new_distance_between_medusa < cur_dis: # 상하좌우로 우선순위 선택 
                                move_cnt += 1 
                                q.append((ny, nx, new_distance_between_medusa))
                                # warrior.loc = (ny, nx) # update
                                war_ys[war_id] = ny 
                                war_xs[war_id] = nx  
                                flag = 1 
                                visited[war_id].append((ny, nx))
                                break # 찾으면 for loop 종료 
                    if not flag:
                        break # 현재 자리에서 이동할 것이 없으면 나와야함. 
                     
                    
                # XXX: 
                if move_cnt > 0: # 움직였으면, 데이터 업데이트 
                    # 새로운 위치 업데이트 (더함)
                    warriors_graph[(war_ys[war_id], war_xs[war_id])].append(war_id)
                    # 기존 위치 (w_y, w_x) 삭제
                    # XXX: 여기서 삭제하지 않으면 lazy deletion 필요 
                    # XXX: O(1)로 삭제할 수 있으면 가능. 근데, 현재 war_id가 warriors_graph[(y, x)] 의 list 어디에 존재하는지 알아야 O(1)으로 삭제 가능 
                    warriors_graph[(w_y, w_x)].remove(war_id)

                total_steps += move_cnt 


            # step4.: 메두사와 같은 칸에 도달한 전사들은 공격 후 사라짐. 
            num_removed_war = len(reached_warriors)
            for removed_war_id in reached_warriors:
                alive[removed_war_id] = False 
                warriors_graph[(war_ys[removed_war_id], war_xs[removed_war_id])].remove(removed_war_id)

            
            # print(total_steps, total_stoned_warriors, num_removed_war)
            print(total_steps, num_stoned_warriors, num_removed_war)
            
            # 살아있는 전사가 없으면 while 문을 break하고 0 0 0 혹은 0을 남은 횟수까지 출력함. 
            if sum(1 if alive[war_id] else 0 for war_id in range(M)) == 0:
                break # while break 
        
        # path가 남아있으면 
        while path:
            ny, nx = path.popleft()
            if ny == e_y and nx == e_x:
                print(0)
            else:
                print(0, 0, 0)

    else:
        print(-1) # 메두사의 집으로부터 공원까지 도달하는 경로가 없는 경우 -1 출력 후 프로그램 종료 


