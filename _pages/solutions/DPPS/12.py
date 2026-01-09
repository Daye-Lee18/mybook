import sys 
from collections import defaultdict , deque 

# sys.stdin = open('Input.txt')
input = sys.stdin.readline

Q = int(input())

'''
필요한 자료구조 
'''
direct_parents = dict()
colors = dict()
max_depths = dict()
direct_childrens = defaultdict(list)
possible_depths = dict()
ROOT = []  # tree전체의 Root node , 여러개 일 수 있음. 점수 조회를 위해서 필요 
total_point = 0 # 점수 조회시 최종 점수 계산 


def dfs(node):
    global total_point 

    # if not direct_childrens[node]: # leaf node인 경우 
    #     total_point += 1 
    #     return 
    
    visited = set()
    # post-order dfs 
    for child in direct_childrens[node]: # leaf node인 경우 자연스럽게 pass 
        # visited.add(colors[child])
        child_point, child_visited =  dfs(child)
        total_point += child_point
        visited = visited.union(child_visited)

    visited.add(colors[node])
    cur_color_num = len(visited)

    return cur_color_num * cur_color_num, visited # 제곱의 합, 있는 색깔 
    




for q in range(Q):
    cmd = list(map(int, input().rstrip().split()))
    if cmd[0] == 100: # 노드 추가: O(1)
        m_id, p_id, color, max_depth = cmd[1:]

        if p_id == -1:
            ROOT.append(m_id) # NOTE: ROOT 는 하나가 아닐 수 있음. 
        
        # root나 모순이 없는 경우에만 추가 
        if p_id == -1 or possible_depths[p_id] > 1:
            direct_parents[m_id] = p_id 
            colors[m_id] = color 
            max_depths[m_id] = max_depth 
            if p_id != -1:
                direct_childrens[p_id].append(m_id) # linked list 
            if p_id == -1:
                possible_depths[m_id] = max_depth
            else:
                possible_depths[m_id] = min(max_depth, possible_depths[p_id] - 1)

    elif cmd[0] == 200: # 색깔 변경 
        m_id, color_changed = cmd[1:]

        q = deque()
        q.append(m_id)

        while q:
            cur_id = q.popleft()

            colors[cur_id] = color_changed # 색깔 변경 

            q.extend(direct_childrens[cur_id])

    elif cmd[0] == 300:
        m_id = cmd[1]
        print(colors[m_id])
    
    else: # cmd[0] == 400: 점수 조회
        # print('a') 
        total_point = 0
        for root in ROOT:
            root_point, _ = dfs(root)
            total_point += root_point 
        print(total_point)

