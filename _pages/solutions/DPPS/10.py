'''
install cameras on the tree nodes where each camera at a node can monitor its parent, itself, 
and its immediate children 
- Binary Tree에서는 총 4개 (parent, itself, left, right)

Return 
- the minmum number of cameras needed to monitor all nodes of the tree 

Constraints 
- 1<= the number of nodes <= 1000
- Node.val == 0  

Idea
-  when deciding to place a camera at a node, 
we might have placed cameras to cover some "subset of this node", its left child, and its right child already. 
- `solve(node)` function returns -> how many cameras it takes to cover the subtree at this node in various states. 
- covered vs. camera placement 
- 3 states for dp table 
    - Assumption: All the nodes below this node are covered 
    [State 0] Strict subtree: The current node is not covered = 부모가 나중에 덮어줄 수 있다. 
    [State 1] Normal subtree: The current node is covered but there is no camera here (on the node)
    [Stete 2] Placed camera: The current node is covered but there is a camera here (which may cover nodes above this node. -> 위쪽도 커버 가능)

- Once we frame the problem in this way, the answer falls out: 
    - To cover a strict subtree, the children of this node must be in state 1. 
    - To cover a normal subtree without placing a camera here,
        - the children of this node must be in states 1 or 2, and at least one of those children must be in state 2 
    - To cover the subtree when placing a camera here, 
        - the children can be in any state.
'''

from typing import Optional, List

# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    def minCameraCover(self, root: Optional[TreeNode]) -> int:
        
        def solve(node: Optional[TreeNode], parent: Optional[TreeNode]):
            if not node:
                return 0, 0, float('inf') # 불가능한 상태를 INF로 저장함으로써, 최종값으로 선택되지 못하도록 함.
            
            left = solve(node.left, node)
            right = solve(node.right, node)

            dp0 = left[1] + right[1] # state 0
            dp1 = min(left[2] + min(right[1:]), min(left[1:])+ right[2]) # state 1 
            dp2 = min(left) + min(right) + 1 # state 2 
            
            return dp0, dp1, dp2
        
        dp0, dp1, dp2 = solve(root, None)
        # return min(max(1, dp[root][1]), max(1,dp[root][2]))
        return min(dp1, dp2)
    
def build_tree(arr: List[int]) -> Optional[TreeNode]:
    from collections import deque 

    if not arr:
        return None 
    
    root = TreeNode(arr[0])
    q = deque()
    q.append(root)

    idx = 1 
    while idx < len(arr):
        node = q.popleft()

        if idx < len(arr) and arr[idx] != None:
            node.left = TreeNode(arr[idx])
            q.append(node.left)
        idx += 1 
        if idx < len(arr) and arr[idx] != None:
            node.right = TreeNode(arr[idx])
            q.append(node.right)
        idx += 1 
    
    return root 


arr = [0] # 1 
# arr = [0, 0] # 1
# arr = [0, 0, None, 0, 0] # 1 
# arr = [0, 0, None, 0, None, 0, None, None, 0] # 2
# arr = [0,0,None,None,0,0,None,None,0,0] # 2
root = build_tree(arr)
sol = Solution()
print(sol.minCameraCover(root))
