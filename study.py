import sys
sys.setrecursionlimit(1_000_000) # 대부분의 경우 10^6으로 하면 무난하게 통과 
input = sys.stdin.readline

MAXN = 100000 + 5

children = [[] for _ in range(MAXN)]
parent   = [-1]*MAXN
color    = [0]*MAXN
maxd     = [0]*MAXN # 현재 노드에서 가질 수 있는 subtree의 최대 depth (간선의 개수)
depth    = [0]*MAXN # 현재 depth: 지금까지의 노드 개수 총합
limitd   = [0]*MAXN # root기준에서 노드 u의 subtree가 도달할 수 있는 최대 허용 깊이
exists   = [False]*MAXN   # 노드 존재 여부

roots = []  # 여러 개의 루트 가능 (forest)

def add_node(mid, pid, c, md):
    # pid == -1: 새 루트
    if pid == -1:
        exists[mid] = True
        parent[mid] = -1
        color[mid]  = c
        maxd[mid]   = md
        depth[mid]  = 0
        limitd[mid] = depth[mid] + maxd[mid]
        roots.append(mid)
        return

    # 부모가 반드시 존재한다고 가정
    dnew = depth[pid] + 1
    # 조상 제약 반영된 최대 허용 깊이와 비교
    if dnew <= limitd[pid]:
        exists[mid] = True
        parent[mid] = pid
        color[mid]  = c
        maxd[mid]   = md
        depth[mid]  = dnew
        limitd[mid] = min(limitd[pid], depth[mid] + maxd[mid])
        children[pid].append(mid)
    # else: 추가 실패 (무시)

def paint_subtree(m, c):
    # iterative DFS로 색칠
    stack = [m]
    while stack:
        u = stack.pop()
        color[u] = c
        for v in children[u]:
            stack.append(v)

def score_once():
    # 비트마스크 DFS: popcount(mask)^2 합
    ans = 0
    # 루트가 여러 개일 수 있어 모두 순회
    for r in roots:
        if not exists[r]:
            continue
        # 포스트오더 한 번
        stack = [(r, 0)]
        order = []
        while stack:
            u, state = stack.pop()
            if state == 0:
                stack.append((u, 1))
                for v in children[u]:
                    stack.append((v, 0))
            else:
                order.append(u)

        mask = [0]*(MAXN)  # 필요부분만 쓰이지만 간단히
        for u in order:
            m = 1 << (color[u]-1)   # 색 1..5 → 0..4
            for v in children[u]:
                m |= mask[v]
            mask[u] = m
            cnt = m.bit_count()
            ans += cnt*cnt
    return ans

Q = int(input())
for _ in range(Q):
    cmd, *rest = map(int, input().split())
    if cmd == 100:
        mid, pid, c, md = rest
        add_node(mid, pid, c, md)
    elif cmd == 200:
        mid, c = rest
        paint_subtree(mid, c)
    elif cmd == 300:
        mid = rest[0]
        print(color[mid])
    else: # 400
        print(score_once())
