'''
entrance: root 
- binary tree 
- automatic contact to the polic 
    - if two directly-linked houses were broken into on the same night 
- given the root of the binary tree, return the maximum amount of money the thief can rob without alerting the police 

Idea 
- postorder DFS, children을 모두 계산한 후에 현재 그 children의 parent를 계산 
- parent를 rob한 경우: children 반드시 들르면 안됨. 
- parent를 rob하지 않은 경우: children의 rob/unrob 상관없음. 


'''

from typing import Optional 

# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    def rob(self, root: Optional[TreeNode]) -> int:
        
        def dfs(node): # post-order 
            if not node:
                return 0, 0
            
            
            left_wo, left_w = dfs(node.left)
            right_wo, right_w = dfs(node.right)

            # 현재 노드 계산 
            cur_node_wo = max(left_wo, left_w) + max(right_wo, right_w)
            cur_node_w = left_wo + right_wo + node.val 
            return cur_node_wo, cur_node_w
        
        wo_best, w_best = dfs(root)

        return max(wo_best, w_best)
    
def build_tree(arr):
    from collections import deque 
    if not arr:
        return None 
    
    root = TreeNode(arr[0])
    q = deque()
    q.append(root)
    idx = 1 
    while idx < len(arr):
        node = q.popleft()
        if idx < len(arr) and arr[idx]:
            node.left = TreeNode(arr[idx])
            q.append(node.left)
        idx += 1 
        if idx < len(arr) and arr[idx]:
            node.right = TreeNode(arr[idx])
            q.append(node.right)
        idx += 1 
    return root 

# arr = [3, 2, 3, None, 3, None, 1] # 7
# arr = [3, 4, 5, 1, 3, None, 1] # 9
# arr = [1] # 1 
# arr = [1, 100, 2] # 102
arr = [4, 1, None, 2, None, 3] # 7
root = build_tree(arr)

sol = Solution()
print(sol.rob(root))