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