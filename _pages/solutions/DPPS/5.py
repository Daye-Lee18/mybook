from typing import List 
'''
1. binary tree가 아니라, k-ary tree -> graph로 연결된 노드들을 for loop으로 다음노드로 넘어가기 
2. parent정보만 알고, 다른 어떤 노드에 연결되어있는지 주어지지 않음. -> k_ary_tree adjacency graph 그리기 
'''

class Solution:
    def longestPath(self, parent: List[int], s: str) -> int:
        n = len(parent)
        g = [[] for _ in range(n)]
        for i, p in enumerate(parent):
            if p == -1:
                continue
            g[p].append(i)
            g[i].append(p)

        self.ans = 1

        def dfs(u: int, p: int) -> int:
            # u에서 시작하는, 유효한 하향 경로 중 1등, 2등 길이
            longest1 = 1  # 가장 긴 것
            longest2 = 1  # 두 번째 긴 것

            for v in g[u]: # O(N)
                if v == p:
                    continue
                child_len = dfs(v, u)

                if s[v] == s[u]:
                    continue  # 문자 같으면 쓸 수 없음

                cand = child_len + 1  # u를 포함한 길이

                if cand > longest1:
                    longest2 = longest1
                    longest1 = cand
                elif cand > longest2:
                    longest2 = cand

            # u를 가운데로 하는 경로
            self.ans = max(self.ans, longest1 + longest2 - 1)

            # 부모에게는 u에서 내려가는 최장 경로 하나만 넘김, 현재 노드 u에서 최장 경로
            return longest1

        dfs(0, -1)
        return self.ans

# parent = [-1, 0, 0, 1, 1, 2, 2]; s = "abccbef" # 5
# parent = [-1,0,0,1,1,2]; s = "abacbe"  # 3 
# parent = [-1,0,0,0]; s = "aabc"  # 3 
# parent = [-1, 0, 1]; s = "aab"  # 2
# 아래 예시의 답: 17
parent = [-1,137,65,60,73,138,81,17,45,163,145,99,29,162,19,20,132,132,13,60,21,18,155,65,13,163,125,102,96,60,50,101,100,86,162,42,162,94,21,56,45,56,13,23,101,76,57,89,4,161,16,139,29,60,44,127,19,68,71,55,13,36,148,129,75,41,107,91,52,42,93,85,125,89,132,13,141,21,152,21,79,160,130,103,46,65,71,33,129,0,19,148,65,125,41,38,104,115,130,164,138,108,65,31,13,60,29,116,26,58,118,10,138,14,28,91,60,47,2,149,99,28,154,71,96,60,106,79,129,83,42,102,34,41,55,31,154,26,34,127,42,133,113,125,113,13,54,132,13,56,13,42,102,135,130,75,25,80,159,39,29,41,89,85,19]
s = "ajunvefrdrpgxltugqqrwisyfwwtldxjgaxsbbkhvuqeoigqssefoyngykgtthpzvsxgxrqedntvsjcpdnupvqtroxmbpsdwoswxfarnixkvcimzgvrevxnxtkkovwxcjmtgqrrsqyshxbfxptuvqrytctujnzzydhpal"
sol = Solution()
print(sol.longestPath(parent, s))
