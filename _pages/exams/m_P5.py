# p5.py — Find the City With the Smallest Number of Neighbors at a Threshold Distance
from typing import List
import math

class Solution:
    def findTheCity(self, n: int, edges: List[List[int]], distanceThreshold: int) -> int:
        '''
        bidirectional 
        '''
        # step 1. floyd warshall algorithm - dp
        INF = int(1e9)
        dp = [[INF]*n for _ in range(n)]

        for i in range(n):
            dp[i][i] = 0

        graph = [[] for _ in range(n)]
        for edge in edges:
            # bidirectional 
            graph[edge[0]].append((edge[1], edge[2]))
            graph[edge[1]].append((edge[0], edge[2]))
            # dp INIT 
            dp[edge[0]][edge[1]] = edge[2]
            dp[edge[1]][edge[0]] = edge[2]

        '''
        floyd-warshall 
        distanceThreshold보다 낮은 path중, he city with the smallest number of cities that are reachable (해당 노드로의 path가 가장 적은 시티) -> 동일한 것이 있으면 시티의 node number가 가장 큰 것을 리턴 
        '''
        for k in range(n):
            for a in range(n):
                for b in range(n):
                    dp[a][b] = min(dp[a][b], dp[a][k] + dp[k][b])

        max_dis = -1
        city_cnt = [0]*n
        res_cities = []
        for i in range(n):
            for j in range(n):
                if 0 < dp[i][j] <= distanceThreshold:
                    city_cnt[i] += 1  
        # print(city_cnt)
        min_cnt = INF
        res_city = -1
        # 거꾸로 세서 동률일 경우 city number가 가장 큰 도시를 return하도록 하기 
        for city in range(n-1, -1, -1):
            if min_cnt > city_cnt[city]:
                min_cnt = city_cnt[city]
                res_city = city 
        return res_city


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
        (5, [[0,1,1],[1,2,10],[2,3,1],[3,4,1],[0,4,10]], 2, 1),
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
