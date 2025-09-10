"""
제한사항 
5 <= N <= 100 
BFS -> O(2N) 
# 장애물이 있으면 왼쪽/위쪽으로 돌아가야하므로 4방향 모두 탐색해야함. 
# 회전도 양방향 회전 전부 고려해야함. 
"""
from collections import deque 

def solution(board):
    n = len(board)
    # 1) 패딩으로 경계 간단히
    B = [[1]*(n+2)]
    for r in range(n):
        B.append([1] + board[r] + [1])
    B.append([1]*(n+2))

    # 시작 상태: (1,1)-(1,2)  (패딩 기준 좌표)
    start = ((1,1),(1,2))

    def neighbors(p1, p2):
        (y1,x1),(y2,x2) = p1, p2
        cand = []

        # 2) 4방향 평이동
        dirs = [(-1,0),(1,0),(0,-1),(0,1)]
        for dy,dx in dirs:
            ny1,nx1 = y1+dy, x1+dx
            ny2,nx2 = y2+dy, x2+dx
            if B[ny1][nx1]==0 and B[ny2][nx2]==0:
                cand.append(((ny1,nx1),(ny2,nx2)))

        # 3) 회전 (가로/세로 각각 양방향)
        if y1 == y2:  # 가로
            for d in [-1,1]:  # 위/아래로 회전
                if B[y1+d][x1]==0 and B[y2+d][x2]==0:
                    # 왼쪽 블록 기준 회전
                    cand.append(((y1, x1), (y1+d, x1)))
                    # 오른쪽 블록 기준 회전
                    cand.append(((y2, x2), (y2+d, x2)))
        elif x1 == x2:  # 세로
            for d in [-1,1]:  # 좌/우로 회전
                if B[y1][x1+d]==0 and B[y2][x2+d]==0:
                    # 위쪽 블록 기준 회전
                    cand.append(((y1, x1), (y1, x1+d)))
                    # 아래쪽 블록 기준 회전
                    cand.append(((y2, x2), (y2, x2+d)))

        # 상태 정규화 (작은 좌표가 앞)
        norm = []
        for a,b in cand:
            if a <= b:
                norm.append((a,b))
            else:
                norm.append((b,a))
        return norm

    # 4) BFS
    q = deque()
    start = tuple(sorted(start))
    q.append((start, 0))
    visited = {start}

    goal = (n, n)  # 패딩 전 기준의 (N-1,N-1)는 패딩 좌표로 (n,n)

    while q:
        (p1, p2), t = q.popleft()
        if p1 == goal or p2 == goal:
            return t
        for nxt in neighbors(p1, p2):
            if nxt not in visited:
                visited.add(nxt)
                q.append((nxt, t+1))

    return -1  # 도달 불가

if __name__ == '__main__':
    board = [[0, 0, 0, 1, 1],[0, 0, 0, 1, 0],[0, 1, 0, 1, 1],[1, 1, 0, 0, 1],[0, 0, 0, 0, 0]] # 7
    # board = [[0, 0], [0, 0]] # 1
    # board = [[0, 0,0], [0, 0,0], [0, 0,0]] # 3
    print(solution(board))