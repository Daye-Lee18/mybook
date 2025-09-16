from collections import deque 

def yx_from_rc(c, r):
    return (N-r-1, c)

def put_new_microbe(y, x, t, v, micro_id):

    # # 바뀐 것에 대해서 작은 것 ~ 큰 것으로 분류 
    # (y, x)는 포함 (t, v)는 불포함 
    # small_y = y if y < t else t 
    # big_y = y if y > t else t 
    # small_x = x if x < v else v 
    # big_x = x if x > v else v 


    print(f'Cur graph after put at {y, x} ~ {t, v}')
    # inclusive ~ exclusive 
    eatened_microbe_ids = set()
    for a in range(y, t, -1):
        for b in range(v-1, x-1, -1):
            if graph[a][b] != 0:
                eatened_microbe_ids.add(graph[a][b])
            # 다른 미생물들은 잡아먹음 
            graph[a][b] = micro_id

    # 기존 무리의 영역이 2개 이상이되면, 나눠진 미생물은 모두 사라짐. 
    # 잡아먹힌 micro_ids에 대해서 확인 
    for removed_id in eatened_microbe_ids:
        nums, locs = count_group_of_id(removed_id) # int, list 
        # 해당 미생물을 없앰 
        if nums >= 2:
            for ry, rx in locs:
                graph[ry][rx] = 0

def in_range(y, x):
    return 0<=y <N and 0<=x <N

def count_group_of_id(m_id):
    cnt = 0
    visited = set()
    DY = [-1, 1, 0, 0]; DX =[0, 0, -1, 1]


    # 맵 전체 돌면서 확인 
    for a in range(N):
        for b in range(N):
            if graph[a][b] == m_id and (a, b) not in visited:
                # visited propgation 
                q = deque([(a, b)])
                visited.add((a, b))
                while q:
                    cury, curx = q.popleft()

                    for t in range(4):
                        ny = cury + DY[t] ; nx = curx + DX[t]
                        if in_range(ny, nx) and (ny, nx) not in visited and graph[ny][nx] == m_id:
                            q.append((ny, nx))
                            visited.add((ny, nx))

                #### BFS가 끝나면 group 1개 
                cnt += 1 
    return cnt, list(visited)

def solve():
    global N, graph
    f = open('/Users/dayelee/Documents/GitHub/mybook/Input.txt', 'r')
    N, Q = map(int, f.readline().strip().split())
    graph = [[0]*N for _ in range(N)]

    for micro_id in range(1, Q+1):
        c1, r1, c2, r2 = map(int, f.readline().strip().split())

        # prelim 
        y, x = yx_from_rc(c1, r1)
        t, v = yx_from_rc(c2, r2)

        # Step 1: 
        # 직사각형 영역에 미생물 투입 [(y, x) ~ (t, v)) inclusive ~ exclusive
        # 다른 미생물들은 잡아먹음 
        # 기존 무리의 영역이 2개 이상이되면, 나눠진 미생물은 모두 사라짐. 
        put_new_microbe(y, x, t, v, micro_id)
        
        for row in graph:
            print(row)

        

if __name__ == '__main__':
    solve()