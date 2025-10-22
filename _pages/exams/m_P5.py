# p5.py — Find the City With the Smallest Number of Neighbors at a Threshold Distance
from typing import List
import math

class Solution:
    def findTheCity(self, n: int, edges: List[List[int]], distanceThreshold: int) -> int:
        # Floyd–Warshall
        INF = 10**15
        dist = [[INF]*n for _ in range(n)]
        for i in range(n):
            dist[i][i] = 0
        for u, v, w in edges:
            dist[u][v] = min(dist[u][v], w)
            dist[v][u] = min(dist[v][u], w)
        for k in range(n):
            dk = dist[k]
            for i in range(n):
                di = dist[i]
                ik = di[k]
                if ik == INF: continue
                for j in range(n):
                    nd = ik + dk[j]
                    if nd < di[j]:
                        di[j] = nd

        best_city = -1
        best_cnt = math.inf  # 최소 개수
        for i in range(n):
            cnt = sum(1 for j in range(n) if i != j and dist[i][j] <= distanceThreshold)
            # 개수 최소, 동률이면 index 큰 도시
            if cnt < best_cnt or (cnt == best_cnt and i > best_city):
                best_cnt = cnt
                best_city = i
        return best_city


def run_tests():
    sol = Solution()
    tests = [
        (4, [[0,1,3],[1,2,1],[1,3,4],[2,3,1]], 4, 3),
        (5, [[0,1,2],[0,4,8],[1,2,3],[1,4,2],[2,3,1],[3,4,1]], 2, 0),
        (2, [[0,1,5]], 4, 1),
        (3, [[0,1,1],[1,2,1],[0,2,1]], 2, 2),
        (6, [[0,1,1],[1,2,1],[2,3,1],[3,4,1],[4,5,1]], 2, 5),
        (3, [[0,1,2],[1,2,2],[0,2,5]], 2, 2),
        (4, [[0,1,1],[1,2,1]], 1, 3),
        (4, [[0,1,1],[0,2,1],[0,3,1],[1,2,1],[1,3,1],[2,3,1]], 1, 3),
        (5, [[0,1,2],[1,2,2],[2,3,2],[3,4,2]], 3, 4),
        # 10: threshold=0 → 모두 이웃 0명, 동률이므로 가장 큰 index
        (3, [[0,1,1],[1,2,1]], 0, 2),
        # 11: 두 컴포넌트 (0-1) (2-3), 모두 1명 → 동률, index 큰 3
        (4, [[0,1,2],[2,3,1]], 2, 3),
        # 12: 중복 간선(더 작은 가중치 반영) → 모두 2명, 동률이므로 2
        (3, [[0,1,5],[0,1,1],[1,2,1]], 2, 2),
        # 13: 스타 그래프, 중심 0 / leaf는 1명 → leaf 동률이므로 4
        (5, [[0,1,1],[0,2,1],[0,3,1],[0,4,1]], 1, 4),
        # 14: 선형 그래프, 큰 threshold로 전체 연결 → 모두 4명, 동률 4
        (5, [[0,1,1],[1,2,1],[2,3,1],[3,4,1]], 10, 4),
        # 15: 0에서만 연결된 별형(가중치 3), thr=3 → leaf들은 1명으로 최소, 동률 3
        (4, [[0,1,3],[0,2,3],[0,3,3]], 3, 3),
        # 16: 비균질 가중치로 일부만 가까움 → 최소 1명 다수, 동률 4
        (5, [[0,1,1],[1,2,10],[2,3,1],[3,4,1],[0,4,10]], 2, 4),
        # 17: n=1 단일 도시 → 이웃 0명, 정답 0
        (1, [], 5, 0),
        # 18: 우회 경로가 직통보다 유리, 최소 2명 동률(0,3) → tie-break로 3
        (4, [[0,1,10],[0,2,1],[2,1,1],[1,3,1],[2,3,10]], 2, 3),
        # 19: 완전그래프(가중치 2) + 한 간선만 1, thr=1 → 2,3은 0명으로 최소 → 3
        (4, [[0,1,1],[0,2,2],[0,3,2],[1,2,2],[1,3,2],[2,3,2]], 1, 3),
        # 20: 간선 없음, 모두 0명 → 동률, 가장 큰 index 3
        (4, [], 100, 3),
    ]
    passed = 0
    for i, (n, edges, thr, expected) in enumerate(tests, 1):
        got = sol.findTheCity(n, edges, thr)
        ok = got == expected
        print(f"[p5][case {i}] n={n}, thr={thr}, edges={edges} -> {got} (expected {expected}) {'OK' if ok else 'FAIL'}")
        passed += ok
    print(f"[p5] Passed {passed}/{len(tests)}")


if __name__ == "__main__":
    run_tests()
