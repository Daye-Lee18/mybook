# Lecture 1-2. 구현 실습  

예시 문제 링크 
- [코드트리 갬빗](https://www.codetree.ai/ko/frequent-problems/samsung-sw/problems/codetrees-gambit/description)
- [4번: 코드트리 민트초코 우유](https://www.codetree.ai/ko/frequent-problems/samsung-sw/problems/mint-choco-milk/description)
- [택배 하차](https://www.codetree.ai/ko/frequent-problems/samsung-sw/problems/delivery-service/description)
- [루돌프의 반란](https://www.codetree.ai/ko/frequent-problems/samsung-sw/problems/rudolph-rebellion/description)
- [왕실의 기사 대결](https://www.codetree.ai/ko/frequent-problems/samsung-sw/problems/royal-knight-duel/description)


## 4번: 민트초코 우유 


````{admonition} solution 
:class: dropdown 

```{code-block} python 
import sys
from collections import deque
import heapq

try:
    sys.stdin = open("input.txt")
except:
    pass
#

#
def print_trust():
    for r in range(N):
        for c in range(N):
            print(board[r][c].trust, end=' ')
        print()
    print('=' * 20)


def print_food():
    for r in range(N):
        for c in range(N):
            print(board[r][c].food, end=' ')
        print()
    print('=' * 20)
#

#
class Student:
    def __init__(self, r, c):
        self.r = r
        self.c = c
        self.food = 0
        self.trust = 0  # 신앙심
        self.defence = 0  # 방어
    #
    def __lt__(self, other):
        return (-self.trust, self.r, self.c) < (-other.trust, other.r, other.c)
#

#
class Group:
    def __init__(self, food):
        # 대표 좌표, 대표 음식
        self.food = food
        self.member = []
        self.manager = None

    #
    def set_manager(self):
        """
        대표자 정하기 & trust set
        대표자: trust up -> r down -> c down
        대표자.trust += 그룹원 -1
        팀원.trust -= 1
        """
        self.manager = min(self.member)
        for m in self.member:
            if m == self.manager:
                m.trust += len(self.member) - 1
            else:
                m.trust -= 1
    #
    def print_group_member(self):
        for m in self.member:
            print((m.r, m.c), end=' ')
        print()
#

#
food2bin = {
    'T': 0,
    'C': 1,
    'M': 2
}
#

#
def in_range(r, c):
    return 0 <= r < N and 0 <= c < N
#

#
def breakfast():
    """
    - 모든 학생의 trust += 1
    """
    for r in range(N):
        for c in range(N):
            stu = board[r][c]
            stu.trust += 1
#

#
dr = [-1, 1, 0, 0]
dc = [0, 0, -1, 1]
def bfs(sr, sc, food, group, v):
    q = deque([(sr, sc)])
    group.member.append(board[sr][sc])
    while q:
        r, c = q.popleft()
        v[r][c] = True
        for i in range(4):
            nr = r + dr[i]
            nc = c + dc[i]
            if not in_range(nr, nc) or v[nr][nc] or board[nr][nc].food != food:
                continue
            q.append((nr, nc))
            v[nr][nc] = True
            group.member.append(board[nr][nc])
    return v
#

#
def lunch():
    """
    - 인접한 학생들과 신봉 음식이 같은 경우 그룹 형성
    - 대표자: trust up -> r down -> c down
    - 대표자.trust += 그룹원 -1
    - 팀원.trust -= 1
    :return:
    """
    visited = [[False] * N for _ in range(N)]
    group_lst = []
    for r in range(N):
        for c in range(N):
            if visited[r][c]:
                continue
            # 그룹 형성
            group = Group(board[r][c].food)
            visited = bfs(r, c, board[r][c].food, group, visited)
            group_lst.append(group)
            # 대표자 정하기 & trust set
            group.set_manager()

    return group_lst
#

#
def set_order(group_lst):
    """
    [1] 전파 순서 정하기
    - 전파 순서: 단일 -> 이중 -> 삼중 그룹
        - 대표자의 B up -> 대표자의 r down -> c down
    :return order_lst
    """
    order_lst = [[] for _ in range(3)]  # 단일/이중/삼중
    for g in group_lst:
        # 삼중인 경우
        if g.food == 7:
            order_lst[2].append(g.manager)
        # 단일 인 경우
        elif g.food in [1, 2, 4]:
            order_lst[0].append(g.manager)
        else:  # 이중인 경우
            order_lst[1].append(g.manager)
    return order_lst
#

#
def strong_spread(x, spreader, target):
    target.food = spreader.food
    x -= (target.trust + 1)
    target.trust += 1

    return x


def weak_spread(x, spreader, target):
    for food in [0, 1, 2]:
        if ((spreader.food >> food) & 1 == 1) and ((target.food >> food) & 1 == 0):
            target.food |= (1 << food)
    target.trust += x
    x = 0

    return x


def do_spread(spreader):
    """
    [2] 전파하기
    - 전파자
        - 전파 방향: trust % 4
        - x: B-1(간절함), B = 1check
    - 전파 시작
        - 전파 방향으로 1칸씩 이동하며 전파
        - 격자 밖으로 나가거나 x = 0이 되면 전파 종료
        - 전파 음식이 타겟음식과 같으면 전파 X, 다음 진행
        - 다르면, 전파 수행
    [3] 전파 방법
    - y: 타겟의 trust
    - 강한 전파(x > y):
        - 타겟.food = 전파.food
        - 전파.x -= y+1
        - 타겟.trust += 1
    - 약한 전파 (x <= y):
        - 타겟.food |= 전파자.food   (있는지 확인 필요)
        - 타겟.trust += x
        - 전파.x = 0
    :return:
    """
    d = spreader.trust % 4
    x = spreader.trust - 1
    spreader.trust = 1
    r, c = spreader.r, spreader.c

    for i in range(1, N):
        nr = r + dr[d] * i
        nc = c + dc[d] * i

        if not in_range(nr, nc) or x <= 0:
            break

        target = board[nr][nc]
        if target.food ^ spreader.food:  # 두 음식이 다를 때만 전파 진행
            if x > target.trust:
                x = strong_spread(x, spreader, target)
            else:
                x = weak_spread(x, spreader, target)
            target.defence = t
#

#
def dinner(group_lst):
    # 전파 순서 정하기
    order_lst = set_order(group_lst)

    for order in order_lst:
        order.sort()
        for spreader in order:
            if spreader.defence < t:
                do_spread(spreader)
#

#
def print_result():
    """
    TCM, TC, TM, CM, M, C, T 순으로 각 음식의 신봉자들의 신앙심 총합 출력
    'T': 0,
    'C': 1,
    'M': 2
    """
    result = [0]*8
    for r in range(N):
        for c in range(N):
            stu = board[r][c]
            result[stu.food] += stu.trust

    for food in ['TCM', 'TC', 'TM', 'CM', 'M', 'C', 'T']:
        key_food = 0
        for f in food:
            key_food |= 1 << food2bin[f]
        print(result[key_food], end=" ")
    print()
#

#
N, T = map(int, input().split())
board = [[0] * N for _ in range(N)]
for row in range(N):
    tmp = input()
    for column in range(N):
        stu = Student(row, column)
        food = food2bin[tmp[column]]
        stu.food |= (1 << food)
        board[row][column] = stu
#
for row in range(N):
    tmp = list(map(int, input().split()))
    for column in range(N):
        board[row][column].trust = tmp[column]
#

#
for t in range(1, T + 1):
    breakfast()

    GROUP_LST = lunch()

    dinner(GROUP_LST)

    print_result()
```
````

````{admonition} 틀린 답 
:class: dropdown 

어디서 틀렸나? 사실 잘 모르겠음. 

중요한 건, 보통의 경우는 list.sort()를 사용하고 아래와 같은 부득이한 경우에만 min-heap을 쓴다. 왜냐면 min-heap의 pop()이후 다시 heapify하는데 시간이 걸리기 때문. 

- 간선이 스트리밍으로 들어오거나 한 번에 다 만들기 어려운 상황(외부 입력/온라인 처리)
- “가장 싼 간선부터 일부만” 처리하며 중간에 조기 종료가 확실한 특수 케이스

```{code-block} python 
import sys 

sys.stdin = open('Input.txt', 'r')

from collections import deque 
import heapq 

N, T = map(int, input().split())
graph = []
belive_graph = []
for n in range(N):
    graph.append(list(input()))

for _ in range(N):
    belive_graph.append(list(map(int, input().split())))

# 간절함 그래프 
desperate_graph =[[-1] * N for _ in range(N)]

def morning():
    for y in range(N):
        for x in range(N):
            belive_graph[y][x] += 1 

def lunch():
    '''
    1. 인접한 학생들과 신봉 음식이 "완전히 같은 경우"에 그룹 형성 
    2. 대표자 선정 
        - 신앙심이 가장 큰 사람 -> y가 작은 사람 -> x가 작은 사람 
    3. 그룹의 대표자에게 <- 그룹 다른 학생들의 신앙심이 넘어감.
        - 대표자는 +1, 그룹 내 다른 학생들 -1 
    '''
    visited = [[False]*N for _ in range(N)]
    # group을 넣을때 순서가 있음 
    groups = []
    for y in range(N):
        for x in range(N):
            if not visited[y][x]:
                locs = BFS(y, x, visited) # 음식순서 -> 대표자가 맨 앞에 있는 list반환, (len(음식), -belive_graph[ny][nx], ny, nx)
                heapq.heappush(groups, locs)

    # 각 그룹을 돌면서 대표자(맨 처음)에게 신앙심 넘겨주기 
    for group in groups:
        # 대표자 
        representative = group[0]
        for other in group[1:]:
            belive_graph[other[2]][other[3]] -= 1
            belive_graph[representative[2]][representative[3]] += 1

    return groups 

def in_range(y, x):
    return 0 <= y < N and 0<=x <N 

def BFS(cury, curx, visited):
    DY = [-1, 0, 1, 0]; DX = [0, 1, 0, -1]
    locs = [(len(graph[cury][curx]),-belive_graph[cury][curx],cury, curx)]
    q = deque([(cury, curx)])
    original_food = graph[cury][curx]
    visited[cury][curx] = True 

    while q:
        y, x = q.popleft()

        for t in range(4):
            ny = y + DY[t]
            nx = x + DX[t]
            # graph내에 있고, 신봉 음식 이름이 원래 (cury, curx)와 동일하고 visited안했다면
            if in_range(ny, nx) and graph[ny][nx] == original_food and not visited[ny][nx]:
                visited[ny][nx]=True 
                q.append((ny, nx))
                # 음식순서: 단일 -> 2개 -> 3개 (min-heap), 대표자: 신앙심이 가장 큰 사람 (max_heap) -> y가 작은 사람 (min-heap) -> x가 작은 사람 (min-heap)
                heapq.heappush(locs, (len(graph[ny][nx]), -belive_graph[ny][nx], ny, nx))

    return locs



def dinner(groups):
    '''
    groups안의 각 group은 다음과 같이 정렬 : heap이라서 heappop()으로 빼야함. 
    # (음식조합개수, -belive_graph[ny][nx], y, x): 음식조합개수 min, 신앙심이 max, y가 최소, x가 최소 

    대표자(==전파자) 의 신앙심 전파
    - 만약 전파하기 전에, 다른 전파자에게 전파 당한 경우 
        - 그 날 전파하지 못함. 
        - 추가로 전파를 받을 수는 있음. 
    - 자신과 다른 음식을 좋아하는 학생이 있는 경우에 전파 방향으로 전파함.
        - 전파 방향: 신앙심 % 4
        - 0: 위, 1: 아래, 2: 왼쪽, 3: 오른쪽
        - 전파자 update 
            - 간절함: 신앙심-1, 신앙심: 1 
    - 전파 방향으로 한 칸 씩 이동하면 전파 
        - 1. 전파 대상의 신봉음식이 동일한 경우, 전파하지 않고 지나감 
        - 2. 신봉음식이 다른 경우 
            - 강한 전파: 전파자의 간절함 > 전파대상의 신앙심인 경우 
                - 전파대상의 신봉 음식 = 전파자의 신봉 음식 
                - 전파자 간절함: 간절함 - (전파대상의 신앙심 + 1)
                    - 간절함이 0이 되면 전파 종료 
                - 전파대상 신앙심: 전파대상의 신앙심 + 1  
            - 약한 전파: 전파자의 간절함 <= 전파대상의 신앙심인 경우 
                - 전파대상의 신봉 음식 += 전파자의 신봉 음식 
                - 전파자 간절함 = 0
                - 전파대상의 신앙심 += 전파자의 간절함 
        - 종료: Graph 밖을 벗어나거나, '간절함' 0이 되면 종료 
    '''
    DY = [-1, 1, 0, 0]; DX = [0, 0, -1, 1]
    is_propagated = [[False] * N for _ in range(N)]

    # 모든 대표자의 위치 
    all_representatives = set()
    for group in groups:
        # 대표자 
        representative = group[0]
        r_y = representative[2]
        r_x = representative[3]
        all_representatives.add((r_y, r_x))


    # 이후 ordered 순서로 전파 실행
    # for representative in ordered:
    #     ...


    # for group in groups: # 순서대로 뽑기 
    while groups:
        group = heapq.heappop(groups)


        # 대표자 
        representative = group[0]
        r_y = representative[2]
        r_x = representative[3]
        # print(f'representative: ({representative[2], representative[3]})')

        # 대표자가 전파당한 경우 전파할 수 없음 
        if is_propagated[r_y][r_x]:
            continue    

        # 전파 시작 
        '''
        전파자 update 
            - 간절함: 신앙심-1, 신앙심: 1 
        '''
        dir = belive_graph[r_y][r_x] % 4  # 빙행 설정 
        desperate_graph[r_y][r_x] = belive_graph[r_y][r_x] -1
        belive_graph[r_y][r_x] = 1
        r_food = graph[r_y][r_x]


        cur_y = r_y + DY[dir]
        cur_x = r_x + DX[dir]
        while desperate_graph[r_y][r_x] > 0 and in_range(cur_y, cur_x): # 간절함이 0이 되면 전파 종료 혹은 그래프를 나가면 종료 
            # 전파 대상의 신봉음식이 동일한 경우, 전파하지 않고 지나감 
            if r_food == graph[cur_y][cur_x]:
                cur_y += DY[dir]
                cur_x += DX[dir]
                # print(f'after propgation with repre {r_y, r_x}')
                # for row in belive_graph:
                #     print(row[:])
                continue 

            # 신봉음식이 다른 경우 
            if desperate_graph[r_y][r_x] > belive_graph[cur_y][cur_x]: # 강한전파 
                '''
                - 전파대상의 신봉 음식 = 전파자의 신봉 음식 
                - 전파자 간절함: 간절함 - (전파대상의 신앙심 + 1)
                - 전파대상 신앙심: 전파대상의 신앙심 + 1  
                '''
                graph[cur_y][cur_x] = r_food 
                belive_graph[cur_y][cur_x] += 1
                desperate_graph[r_y][r_x] -= belive_graph[cur_y][cur_x]
            
            else: # 약한 전파 
                '''
                전파대상의 신봉 음식 += 전파자의 신봉 음식 
                - 전파자 간절함 = 0
                - 전파대상의 신앙심 += 전파자의 간절함 
                
                'TCM' -> 'TC' -> 'TM' -> 'CM' -> 'M' -> 'C' -> 'T' 
                민트초코우유 -> 민트초코 -> 민트우유 -> 초코우유 -> 우유 -> 초코 -> 민트 순서대로 각 음식의 신앙심 출력 
                '''
                cur_food = set(graph[cur_y][cur_x])
                cur_food = cur_food.union(set(graph[r_y][r_x]))
                
                result_food = []
                if 'T' in cur_food:
                    result_food.append('T')
                if 'C' in cur_food:
                    result_food.append('C')
                if 'M' in cur_food:
                    result_food.append('M')
                

                graph[cur_y][cur_x] = ''.join(result_food)
                # print(result_food)
                belive_graph[cur_y][cur_x] += desperate_graph[r_y][r_x]
                desperate_graph[r_y][r_x] = 0

            ## update 
            if check_representative_propagaged(cur_y, cur_x, all_representatives):
                is_propagated[cur_y][cur_x] = True 

            cur_y += DY[dir]
            cur_x += DX[dir]

            # print(f'after propgation with repre {r_y, r_x}')
            # for row in belive_graph:
            #     print(row[:])


def check_representative_propagaged(y, x, repres_set):
    return (y, x) in repres_set


def print_believe(idx):
    '''
    'TCM' -> 'TC' -> 'TM' -> 'CM' -> 'M' -> 'C' -> 'T' 
    민트초코우유 -> 민트초코 -> 민트우유 -> 초코우유 -> 우유 -> 초코 -> 민트 순서대로 각 음식의 신앙심 출력 
    '''
    my_dict = {
        'TCM': 0,
        'TC': 1,
        'TM': 2,
        'CM': 3,
        'M': 4,
        'C': 5,
        'T': 6
    }
    believe_sums = [0] * 7

    for y in range(N):
        for x in range(N):
            believe_sums[my_dict[graph[y][x]]] += belive_graph[y][x]

    for i in believe_sums:
        print(i, end=' ')
    if idx != T-1:
        print()

def solve():
    for t in range(T):
        # 아침 
        morning()
        # print('after morning: ')
        # for row in belive_graph:
        #     print(row[:])

        # 점심 
        groups = lunch()
        # print('after lunch belif: ')
        # for row in belive_graph:
        #     print(row[:])
        # print('after lunch food: ')
        # for row in graph:
        #     print(row[:])

        # 저녁 
        dinner(groups)
        # print('after dinner: ')
        # for row in belive_graph:
        #     print(row[:])

        # 출력 
        print_believe(t)

if __name__ == '__main__':
    solve()    
```
````