from collections import deque 


def in_range(y, x):
    return 0<=y <N and 0<=x <N

def is_closer_to_lower_left(tuple1, tuple2):
    if tuple1[0] != tuple2[0]:
        return tuple1[0] > tuple2[0] # y가 더 큰 것 
    return tuple1[1] < tuple2[1] # x는 더 작은 것 

def sort_locs_closer_to_lower_left(tuple_list):
    tuple_list_len = len(tuple_list)
    for f in range(tuple_list_len):
        lowest = f 
        for r in range(f+1, tuple_list_len):
            if not is_closer_to_lower_left(tuple_list[lowest], tuple_list[r]):
                lowest = r 
        
        if lowest != f:
            # SWAP 
            temp_tuple = tuple_list[lowest]
            tuple_list[lowest] = tuple_list[f]
            tuple_list[f] = temp_tuple

def move_microbe(new_graph):
    # 현재 global graph에 있는 미생물들을 보고 new_graph에 옮겨 담기 
    # Step 1: 1개의 그룹씩 있는 각 미생물들의 위치 locs구하기, len(locs)이 가장 높은 것이 가장 먼저 옮겨짐 
    # 모양은 원점에서 가장 왼쪽 아래 (바꾼 그래프 상으로는 왼쪽 위)와 가장 가까운 점까지의 거리만큼 모두 이동하면 됨.
    DY = [-1, 1, 0, 0]; DX =[0, 0, -1, 1]
    id_to_locs = dict()
    
    # 현재 최대 cur_micro_id만큼의 개수가 존재할 수 있음 (지워진 것 빼고)
    for a in range(N):
        for b in range(N):
            
            if graph[a][b] != 0 and graph[a][b] not in id_to_locs:
                locs = set()  # NOTE: locations 추적, 새로운 ID 발견할 때마다 새롭게 INIT 필요 
                q = deque([(a, b)])
                locs.add((a, b))
                while q:
                    cury, curx = q.popleft()
                    for t in range(4):
                        ny = cury + DY[t] ; nx = curx + DX[t]
                        # 현재 micro_idx와도 동일해야함 
                        if in_range(ny, nx) and (ny, nx) not in locs and graph[ny][nx] == graph[a][b]:
                            q.append((ny, nx))
                            locs.add((ny, nx))
                
                # micro_id마다 위치를 연결하는 dictionary 생성 
                id_to_locs[graph[a][b]] = locs

    print(f'Current id with locs \n{id_to_locs}')
    print('Len: ', len(id_to_locs))

    # id_to_locs을 정렬 
    # 1. 개수가 가장 많은 것
    # 2. 개수가 동일하다면 가장 먼저 투입된 미생물 선택 
    id_to_locs_list = []
    for key, value in id_to_locs.items():
        id_to_locs_list.append((key, value))

    sort_list(id_to_locs_list) # call by reference 
    print(f"Sorted as described in the problem \n {id_to_locs_list}")
    
    # locs은 origin (7, 0) 과 가장 가까운 순서대로 정렬 즉, ascending order 
    for idx, (id, locs) in enumerate(id_to_locs_list):
        # set 재할당 
        # new_locs = sort_locs_closer_to_lower_left(locs)
        locs = list(locs)
        sort_locs_closer_to_lower_left(locs)
        id_to_locs_list[idx] = (id, locs)

    print(f"Sorted id_to_locs closer to the origin \n {id_to_locs_list}")

    each_nums = [] # 옮겨서 살아남은 미생물들의 개수만 append 

    # 옮기기 
    # 모든 미생물에 대해 
    for id, locs in id_to_locs_list:
        is_ok = True 
        done = False 
        # 이 조건안에서 colum 좌표가 작은 위치로 옮기고, 그 위치가 둘 이상이면 최대한 row 좌표가 작은 위치로 오도록 옮김 (x좌표가 작은 위치 -> y가 작은 위치)
        # 시작 위치 찾기: b를 outer loop에 둬서 x가 작은 위치를 먼저 확인  
        for ori_y in range(N-1, -1, -1):
            for ori_x in range(N):
                if new_graph[ori_y][ori_x] != 0:  # 옮길 그래프에서 시작점 for loop들어가기 전에 미리 체크해서 시간 줄이기 
                    continue

                # 미생물 무리가 차지한 영역이 배양 용기를 벗어나지 않아야함 
                # 현재 locs의 모든 점이 (a,b)만큼 평행 이동했을 때 벗어나면 안됨 
                # NOTE: dif_y, dif_x는 제일 작은 지점과의 거리임!! 고정!!!! 
                dif_y = locs[0][0] - ori_y
                dif_x = locs[0][1] - ori_x
                
                for cur_y, cur_x in locs:
                    # NOTE: dif_y, dif_x는 제일 작은 지점과의 거리임!! 
                    n_y = cur_y-dif_y
                    n_x = cur_x-dif_x
                    # print(f"Move point {cur_y, cur_x} -> {n_y, n_x}")
                    # 평행 이동한 점들이 배양용기를 벗어나거나 다른 microbe가 있으면 
                    if (not in_range(n_y,n_x)) or (new_graph[n_y][n_x] !=0):
                        is_ok = False 
                        break 
                
                
                # 다른 미생물이 이미 있으면 이곳을 시작점으로 할 수 없음 
                if is_ok and new_graph[ori_y][ori_x] == 0: # 괜찮으면 new_graph에 옮김 
                    dif_y = locs[0][0] - ori_y
                    dif_x = locs[0][1] - ori_x
                    for cur_y, cur_x in locs:
                        n_y = cur_y-dif_y
                        n_x = cur_x-dif_x
                        new_graph[n_y][n_x] = id
                    
                    done = True 
                    # each_nums update 
                    each_nums.append(len(locs))
                    # 한번 옮겼으면 2중 for loop을 멈춰야함. 

                if done:
                    break  # stack구조에서 가장 안쪽 for loop 
            if done:
                break # stack구조에서 가장 바깥쪽 for loop  
                # 그게 아니라면 어떤 곳에도 둘 수 없다면 사라짐  (update를 안하면 됨.)
    # 살아남은 것들만 개수 return 
    return each_nums 


def compare(locs1, locs2, id1, id2):
    if len(locs1) != locs2:
        return len(locs1) > len(locs2) # 개수가 더 많은 것이 좋음
    if id1 != id2:
        return id1 < id2  # 먼저 들어온 것일수록 좋음 
    else:
        return True 

def sort_list(my_list):
    list_len = len(my_list)

    for front in range(list_len):
        lowest = front 
        for rear in range(front+1, list_len):
            # compare(locs1, locs2, id1, id2)
            if not compare(my_list[lowest][1], my_list[rear][1], my_list[lowest][0], my_list[rear][0]):
                lowest = rear 
        
        # 위치 변경이 있는 경우 SWAP 
        if lowest != front: 
            temp_tuple = my_list[front]
            my_list[front] = my_list[lowest] # tuple의 재할당은 가능 , 다만 tuple은 t[0] = 99로 'item' assignment는 하지못함. 
            my_list[lowest]= temp_tuple



global graph, N

# graph = [
#     [1, 1, 0, 0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 0, 0, 0],
# ]
# graph = [
#     [0, 0, 0, 0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 0, 0, 0],
#     [0, 0, 1, 1, 1, 0, 0, 0],
#     [0, 0, 1, 1, 1, 0, 0, 0],
#     [0, 0, 1, 1, 1, 0, 0, 0],
#     [0, 0, 1, 1, 1, 0, 0, 0],
#     [0, 0, 0, 0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 0, 0, 0],
# ]

# graph = [
#     [0, 0, 0, 0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 0, 0, 0],
#     [0, 0, 1, 1, 1, 0, 0, 0],
#     [0, 0, 2, 2, 2, 0, 0, 0],
#     [0, 0, 2, 2, 2, 0, 0, 0],
#     [0, 0, 2, 2, 2, 0, 0, 0],
#     [0, 0, 2, 2, 2, 0, 0, 0],
#     [0, 0, 2, 2, 2, 0, 0, 0],
# ]
graph = [
    [0, 0, 3, 3, 3, 0, 0, 0],
    [0, 4, 4, 4, 4, 4, 0, 0],
    [0, 4, 4, 4, 4, 4, 0, 0],
    [0, 4, 4, 4, 4, 4, 0, 0],
    [0, 4, 4, 4, 4, 4, 0, 0],
    [0, 4, 4, 4, 4, 4, 0, 0],
    [0, 0, 2, 2, 2, 0, 0, 0],
    [0, 0, 2, 2, 2, 0, 0, 0],
]

print("Original graph:")
for row in graph:
    print(row)
N = len(graph)

new_graph = [[0]*N for _ in range(N)]
each_nums = move_microbe(new_graph)

print("Moved graph")
for row in new_graph:
    print(row)

print(each_nums)


