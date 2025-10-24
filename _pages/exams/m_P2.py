# p2.py — Invert Binary Tree (BFS/DFS 상관없음)

from collections import deque
from typing import Optional, List, Any

class TreeNode:
    def __init__(self, val: int = 0, left: Optional["TreeNode"]=None, right: Optional["TreeNode"]=None):
        self.val = val
        self.left = left
        self.right = right
# DFS 
class Solution:
    def invertTree(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        if not root:
            return root
        root.left, root.right = self.invertTree(root.right), self.invertTree(root.left)
        return root

# BFS 
class Solution:
    def invertTree(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        if not root:
            return None

        q = deque([root])  # BFS용 큐 초기화

        while q:
            node = q.popleft()
            
            # 왼쪽, 오른쪽 자식 교환
            node.left, node.right = node.right, node.left

            # 다음 레벨 탐색
            if node.left:
                q.append(node.left)
            if node.right:
                q.append(node.right)
        
        return root


        
# --- Helpers: level-order <-> list (LeetCode 스타일) ---
def build_tree(vals: List[Optional[int]]) -> Optional[TreeNode]:
    if not vals or vals[0] is None:
        return None
    root = TreeNode(vals[0])
    q = deque([root])
    i = 1
    while q and i < len(vals):
        node = q.popleft()
        if i < len(vals) and vals[i] is not None:
            node.left = TreeNode(vals[i])
            q.append(node.left)
        i += 1 # vals[i]가 None일때도 i를 증가시켜야하므로, if문 밖에 적어준다. 
        if i < len(vals) and vals[i] is not None:
            node.right = TreeNode(vals[i])
            q.append(node.right)
        i += 1 # vals[i]가 None일때도 i를 증가시켜야하므로, if문 밖에 적어준다. 
    return root


def tree_to_list(root: Optional[TreeNode]) -> List[Optional[int]]:
    if not root:
        return []
    res: List[Optional[int]] = []
    q = deque([root])
    while q:
        node = q.popleft()
        if node is None:
            res.append(None)
            continue
        res.append(node.val)
        q.append(node.left)
        q.append(node.right)
    # trim trailing None
    while res and res[-1] is None:
        res.pop()
    return res


def run_tests():
    sol = Solution()
    tests = [
        ([4,2,7,1,3,6,9], [4,7,2,9,6,3,1]),
        ([2,1,3], [2,3,1]),
        ([], []),
        ([1,2,None,3,None], [1,None,2,None,3]),
        ([1,2,3,4,5,None,6], [1,3,2,6,None,5,4]),
        ([1, None, 2], [1,2]),
        ([1,2,None,None,3], [1,None,2,3]),
        ([1,2,3,4,5,6,7], [1,3,2,7,6,5,4]),
        ([1,2,3,None,4,5], [1,3,2,None,5,4]),
        ([1], [1]),  # 10: 단일 노드
        ([1,2,None,3,None,4], [1, None, 2, None, 3, None, 4]),  # 11: 왼쪽으로만 내려가는 사슬 → 오른쪽 사슬
        ([1, None, 2, None, 3, None, 4], [1, 2, None, 3, None, 4]),  # 12: 오른쪽 사슬 → 왼쪽 사슬
        ([0,-1,1,-2,-3,3,2], [0,1,-1,2,3,-3,-2]),  # 13: 음수/양수 섞인 완전 이진
        ([1, None, 2, 3], [1, 2, None, None, 3]),  # 14: 비정형 레벨오더(남는 1개는 왼쪽으로 붙음) → 좌우 반전
        ([4,2,7,1, None, 6, None], [4,7,2, None, 6, None, 1]),  # 15: 중간에 None 포함
        ([1,1,1,1, None, None, 1], [1,1,1,1, None, None, 1]),  # 16: 대칭 구조(불변)
        ([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15],
         [1,3,2,7,6,5,4,15,14,13,12,11,10,9,8]),  # 17: 완전 이진 트리 깊이 3
        ([5,3], [5, None, 3]),  # 18: 왼쪽 자식만 있는 경우 → 오른쪽 자식만
        ([5, None, 3], [5, 3]),  # 19: 오른쪽 자식만 있는 경우 → 왼쪽 자식만
        ([1,2,3, None, None, None, 4], [1,3,2,4]),  # 20: 오른쪽 서브트리의 오른쪽 리프가 왼쪽으로 이동
    ]
    passed = 0
    for i, (inp, expected) in enumerate(tests, 1):
        root = build_tree(inp)
        out_root = sol.invertTree(root)
        got = tree_to_list(out_root)
        ok = got == expected
        print(f"[p2][case {i}] input={inp} -> {got} (expected {expected}) {'OK' if ok else 'FAIL'}")
        passed += ok
    print(f"[p2] Passed {passed}/{len(tests)}")


if __name__ == "__main__":
    run_tests()
