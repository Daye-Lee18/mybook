# p3.py — Binary Tree Paths (Backtracking)

from typing import Optional, List
from collections import deque

class TreeNode:
    def __init__(self, val: int = 0, left: Optional["TreeNode"]=None, right: Optional["TreeNode"]=None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    def binaryTreePaths(self, root: Optional[TreeNode]) -> List[str]:
        if not root:
            return []
        res: List[str] = []
        path: List[str] = []

        def dfs(node: TreeNode):
            path.append(str(node.val))
            if not node.left and not node.right:
                res.append("->".join(path))
            else:
                if node.left:
                    dfs(node.left)
                if node.right:
                    dfs(node.right)
            path.pop()

        dfs(root)
        return res


# Helpers to build tree from LeetCode list
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
        i += 1
        if i < len(vals) and vals[i] is not None:
            node.right = TreeNode(vals[i])
            q.append(node.right)
        i += 1
    return root



def run_tests():
    sol = Solution()
    tests = [
        # 기존 1~5
        ([1,2,3,None,5], ["1->2->5","1->3"]),
        ([1], ["1"]),
        ([1,2,3,4,5,6], ["1->2->4","1->2->5","1->3->6"]),
        ([], []),
        ([1,2,None,3,None,4,None], ["1->2->3->4"]),
        # 추가 6~9
        ([1,2,3,4,None,None,5,6], ["1->2->4->6","1->3->5"]),
        ([1,None,2,3], ["1->2->3"]),
        ([1,2,3,None,None,4,5], ["1->2","1->3->4","1->3->5"]),
        ([1,-2,-3,4], ["1->-2->4","1->-3"]),
        ([1,2,3,4,5,6,7], ["1->2->4","1->2->5","1->3->6","1->3->7"]),     # 10
        ([1,None,2,None,3,None,4], ["1->2->3->4"]),                       # 11
        ([0,-1,2,None,3], ["0->-1->3","0->2"]),                           # 12
        ([1,None,2,3,4], ["1->2->3","1->2->4"]),                          # 13
        ([2,3,None,4,None,5], ["2->3->4->5"]),                             # 14
        ([1,2,3,4,5,None,None,7], ["1->2->4->7","1->2->5", "1->3"]),              # 15
        ([1,2,3,None,None,4,5,6,7], ["1->2", "1->3->4->6","1->3->4->7","1->3->5"]),# 16
        ([5,1,9,None,2,None,3], ["5->1->2","5->9->3"]),                    # 17
        ([2,None,3,None,4,None,5], ["2->3->4->5"]),                        # 18
        ([0], ["0"]),                                                      # 19
        ([1,2,3,4,None,None,None,None,5], ["1->2->4->5","1->3"]),         # 20
    ]
    passed = 0
    for i, (inp, expected) in enumerate(tests, 1):
        root = build_tree(inp)
        got = sol.binaryTreePaths(root)
        # 순서 무관 → 멀티셋 비교
        ok = sorted(got) == sorted(expected)
        print(f"[p3][case {i}] input={inp} -> {got} (expected {expected}) {'OK' if ok else 'FAIL'}")
        passed += ok
    print(f"[p3] Passed {passed}/{len(tests)}")


if __name__ == "__main__":
    run_tests()
