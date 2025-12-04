import sys 
from collections import deque, defaultdict 
from heapq import heappush, heappop 

# sys.stdin = open("Input.txt")
input = sys.stdin.readline

def in_range(x, y):
    global N
    return 0 <= x < N and 0 <= y < N

def BFS(start_x, start_y, cur_id, visited):
    global DX, DY

    visited.add((start_x, start_y))
    q = deque([(start_x, start_y)])

    while q:
        cur_x, cur_y = q.popleft()

        for t in range(4):
            nxt_x = cur_x + DX[t]
            nxt_y = cur_y + DY[t]

            if in_range(nxt_x, nxt_y) and not (nxt_x, nxt_y) in visited:
                if graph[nxt_x][nxt_y] == cur_id:
                    q.append((nxt_x, nxt_y))
                    visited.add((nxt_x, nxt_y))

            
def count_group_bfs(cur_id):
    global graph 
    cnt = 0
    visited = set()
    for y in range(N):
        for x in range(N):
            if graph[x][y] == cur_id and not (x, y) in visited:
                BFS(x, y, cur_id, visited)
                cnt +=1 
    return cnt 

def add(r1, c1, r2, c2, id):
    global graph, does_id_exist
    removed_set = set()
    # Step1: 새롭게 투입된 미생물 graph에 삽입 
    for x in range(r1, r2): # upper right : exclusive 
        for y in range(c1, c2):
            if graph[x][y] != 0:
                removed_set.add(graph[x][y])
            graph[x][y] = id

    does_id_exist.append(True) # 새로운 id있으면 삽입 
    # lower_bottom_coord[id] = (r1, c1)
    
    # Step 2: 앞선 지워진 기존 미생물의 그룹 수가 2개 이상인지 확인 
    for is_removed_id in removed_set:
        cnt = count_group_bfs(is_removed_id)
        # NOTE: 그룹 수가 아예없을 때도 False로 바꾸어주어야한다. 
        if  cnt >= 2 or cnt == 0:
            does_id_exist[is_removed_id] = False 
            # 기존에 있던 location의 정보 다 삭제 

def move():
    global N, id_to_locs, DX, DY, graph, does_id_exist, id_to_new_graph_origin
    new_graph = [[0] * (N) for _ in range(N)]
    sort_pq = []
    id_to_locs = defaultdict(list) # NOTE: 이전과 헷갈리지 않게 reinit 

    # Stpe1. 각 id에 대해서 Locs에 대해 조사 
    visited = set()
    for y in range(N):
        for x in range(N):
            # 현재 아이디 = graph[x][y]에 대해서 방문하지 않았던 것이면 
            # 또한, add()함수에서 지워진, 미생물에 대해서 다시 0으로 만들지 않았으므로 
            # does_id_exist[id]에 대해서도 현재 살아있는 것인지 lazy validtaion을 진행한다. 
            cur_id = graph[x][y] 
            if cur_id != 0 and (x, y) not in visited and does_id_exist[cur_id]:
                # cur_id = graph[x][y]
                cnt = 0
                q = deque([(x, y)])
                visited.add((x, y))

                while q:
                    cur_x, cur_y = q.popleft()
                    # 새로운 것을 만날 때마다 
                    cnt += 1 
                    # 예전 미생물이 다른 미생물한테 어느 정도 먹혀서 제일 작은 coordinate이 다를 수도 있음. 
                    heappush(id_to_locs[cur_id], (cur_x, cur_y)) # min_heap, x와 y모두 
                    # id_to_locs[id].append((cur_x, cur_y)) 

                    for t in range(4):
                        nxt_x = cur_x + DX[t]
                        nxt_y = cur_y + DY[t]

                        if not in_range(nxt_x, nxt_y):
                            continue 
                        if (nxt_x, nxt_y) not in visited and graph[nxt_x][nxt_y] == cur_id:
                            visited.add((nxt_x, nxt_y))
                            q.append((nxt_x, nxt_y))
                # 다 돌았으면 
                heappush(sort_pq, (cnt*-1, cur_id)) # cnt에 대해서는 max heap, cur_id에 대해서 min_heap 

    # Step2. 미생물을 새로운 배양 용기에 옮김 
    while sort_pq:
        cur_area, cur_id = heappop(sort_pq)
        cur_area *= -1 

        locs = id_to_locs[cur_id]
        ref_coord = locs[0]
        
        '''
        NOTE: flag, found를 사용할 때, for loop이나 while loop이 여러개 중첩되어 있는 경우, 
        반드시 "모든 경우의 수"에 대해서 for/while loop이 끝났을 경우, 결과가 어떻게 되는지 확인하여 flag 처리를 각별히 주의해준다. 
        '''
        flag = 0
        # x좌표가 제일 작고, 같은 x위치가 2개이면, y의 위치가 작아야한다. 
        for origin_x in range(N):
            for origin_y in range(N):
                if new_graph[origin_x][origin_y] != 0:
                    continue 
                # 현재 cur_id에 대해 모든 locs에서 미생물을 옮길 수 있는지 확인 
                dif_x = ref_coord[0] - origin_x 
                dif_y = ref_coord[1] - origin_y
                
                found = 1
                for cur_loc in locs:
                    new_x = cur_loc[0] - dif_x
                    new_y = cur_loc[1] - dif_y

                    # 조건1. 배양 용기의 범위를 벗어나지 않아야 함. 
                    if not in_range(new_x, new_y):
                        found = 0
                        break 
                    # 조건 2. 다른 미생물의 영역과 겹치지 않도록 두기 
                    if new_graph[new_x][new_y] != 0:
                        found = 0
                        break 
                # 만약 위의 모든 locs에 대해서 만족하였더라면, 옮길 수 있음 
                if found:
                    id_to_new_graph_origin[cur_id] = (origin_x, origin_y)
                    for cur_loc in locs:
                        new_x = cur_loc[0] - dif_x
                        new_y = cur_loc[1] - dif_y
                        new_graph[new_x][new_y] = cur_id
                        flag = 1 # default로 0으로 해놓고 찾았을 때만 1로 해야함. 만약 default로 1 해놓으면, 아무것도 못찾고 그냥 끝나버림 
                    break # for문 break
                # else:
                #     # 조건을 만족하지 못한 미생물은 사라져야함. 
                #     does_id_exist[cur_id] = False 
            if flag:
                break

        if flag == 0:
            does_id_exist[cur_id] = False 
    # Step 3: 
    graph = new_graph[:]

def record():
    global id_to_locs, does_id_exist, id_to_new_graph_origin, DY, DX
    total = 0

    
    for id in range(1, len(does_id_exist)):
        # 모든 id 에 대하여 인접한 곳에 다른 id 가 있으면, 
        if does_id_exist[id]:
            locs = id_to_locs[id] # move하기 전의 locations 
            ref_coord = locs[0]
            dif_x = ref_coord[0] - id_to_new_graph_origin[id][0]
            dif_y = ref_coord[1] - id_to_new_graph_origin[id][1]
            pair_set = set()

            # 하나의 id에 다른 id가 존재하는 경우 
            for loc_x, loc_y in locs:
                new_x = loc_x - dif_x 
                new_y = loc_y - dif_y 

                for t in range(4):
                    nxt_x = new_x + DX[t]
                    nxt_y = new_y + DY[t]
                    if in_range(nxt_x, nxt_y) and graph[nxt_x][nxt_y] != 0:
                        nxt_id = graph[nxt_x][nxt_y]
                        if (graph[nxt_x][nxt_y] != id) and (not nxt_id in pair_set):
                            if does_id_exist[nxt_id]:
                                pair_set.add(nxt_id)
                                total += len(id_to_locs[id]) * len(id_to_locs[nxt_id])

    print(int(total / 2)) # (A, B) 와 (B, A)는 동일하므로 /2 로 나눠줌. 


N, Q = map(int, input().rstrip().split())

'''
필요한 자료구조 
'''
graph = [[0]*(N) for _ in range(N)]
does_id_exist = [True]
id_to_locs = defaultdict(list) # id_to_locs[id] = [항상 맨 앞에 lower-left loc]
DY = [-1, 1, 0, 0]; DX = [0, 0, -1, 1]
id_to_new_graph_origin = dict()

for id in range(1, Q+1): # microbe_id는 1부터 시작, graph가 0이면 아무것도 없음.
    # if id == 4:
    #     print('a') 
    r1, c1, r2, c2 = map(int, input().rstrip().split())

    add(r1, c1, r2, c2, id)
    move()
    record()