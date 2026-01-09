'''
unrooted tree with `n` nodes
- 0-indexed (0 ~ n-1)
- edges[i] = [a, b]
- each node <-> price 
- price[i] = ith node price 

- the tree can be rooted at any node `root` of your choice. 
- The incurred `cost` after choosing `root` is the difference between the maximum and minimum price sum amongst all paths starting at root. 

Return 
- the maximum possible cost amongst all possible root choices 

Constraints 
- 1 < = n <= 1e5 ~ O(N), O(NlogN)
- edges.length == n - 1
- 0 <= a, b <= n -1
- 1 <= price[i] <= 1e5 : 가격에는 음수가 없음. 최소 1부터 시작 

Idea 
- 원래 DFS는 leaf node로부터 root까지로 가는 path를 계산한다. 그리고 O((N+N-1))의 시간 복잡도를 가지는데, 
- 이 path는 root를 꼭 거쳐야해서, 
- 문제는, root로 가는 길에서의 최대 max sum을 구하는 것이라 모든 노드에 대해서 계산하지 못한다. 
- 따라서, 2번의 dfs를 따로 실행하여 O(N)의 시간 복잡도를 가지게 할 수 있는데, 
    - DFS1() 에서 root 0 의 tree에서 아래로 가는 dp[node][0], dp[node][1]을 만들어서 top 2 의 path sum을 저장
    - DFS2() 에서 각 노드가 root노드가 되도록 re-rooting 했을 때 가질 수 있는 최대값을 갱신한다. 
'''
from typing import List 
import collections 

class Solution:
    def maxOutput(self, n: int, edges: List[List[int]], price: List[int]) -> int:
        res = 0
        dp = [[0, 0] for _ in range(n)]
        graph = collections.defaultdict(list)
        for u, v in edges:
            graph[u].append(v)
            graph[v].append(u)

        # Step 1. Downward DP with the root 0 
        # 0을 루트로 했을 때, 자식으로만 갈 수 있는 최댓값 계산 
        def dfs1(node: int = 0, parent: int = -1) -> int:
            for next_node in graph[node]:
                if next_node == parent: 
                    continue
                v = dfs1(next_node, node) + price[next_node]
                
                # dp[node][0], dp[node][1] : cur_node - next_node .... 에서,"U자신은 제외하고" next_node -> ...의 최대 path sum 2개 저장 
                if v >= dp[node][0]: 
                    dp[node][0], dp[node][1] = v, dp[node][0]
                elif v > dp[node][1]:
                    dp[node][1] = v

            return dp[node][0]

        # Step 2. re-rooting 
        # 부모 쪽 (트리의 나머지 방향)도 하나의 자식처럼 취급해서 각 노드가 루트가 될 때의 최댓값 dp[node][0]을 완성 
        '''
        parent (rooted at 0) -> node -> children 
        -> 
        node - parent (consider it as another child)
             - chldren 
        '''
        def dfs2(node: int = 0, parent: int = -1) -> None:
            if parent != - 1:
                if dp[node][0] + price[node] == dp[parent][0]: # 현재 parent에서 node로 가는 길이 가장 큰 path인 경우 
                    # 현재 node가 root가 될 경우, 가장 큰 path sum의 candidate은
                    # 현재 노드를 포함하지 않은 2등을 사용해서 계산 
                    parent_subtree_max_value = dp[parent][1] + price[parent]
                else: # 현재 parent에서 node로 가는 길이 현재 노드로 가는 길이 아닌 경우, 
                    # 현재 node로 re-rooting할때 가장 큰 값은 현재 node를 포함하지 않은 1등 path를 이용하여 계산 
                    parent_subtree_max_value = dp[parent][0] + price[parent]
                
                # update 
                if parent_subtree_max_value >= dp[node][0]:
                    dp[node][0], dp[node][1] = parent_subtree_max_value, dp[node][0]
                elif parent_subtree_max_value > dp[node][1]:
                    dp[node][1] = parent_subtree_max_value

            for next_node in graph[node]:
                if next_node == parent:
                    continue
                dfs2(next_node, node)

        dfs1() 
        dfs2()

        return max(x[0] for x in dp)
n = 6; edges = [[0,1],[1,2],[1,3],[3,4],[3,5]]; price = [9,8,7,6,10,5]
sol = Solution()
print(sol.maxOutput(n, edges, price))