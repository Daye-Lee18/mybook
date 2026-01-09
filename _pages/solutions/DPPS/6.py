'''
- the path does not need to pass through the root. 
- the path sum of a path is a sum of the node's values in the path 

- Given the `root` of a binary tree, return the maximum path sum of any non-empty path 

Constraints
- 1<= # of nodes <= 3*1e4 
- -1000 <= Node.val <= 1000 
-> node.val은 overlap될 수 있음. 

edge cases 
- node number 1 
- node number 3*1e4 ~ O(N)
- node.val이 모두 양수 
- node.val이 모두 음수 
- node.val이 양수, 음수 섞여 있는 경우 
'''

from typing import Optional 
from collections import defaultdict, deque 

# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    def maxPathSum(self, root: Optional[TreeNode]) -> int:
        
        dp = defaultdict(int)
        max_sum = int(1e9)*-1
        def dfs(node):
            nonlocal max_sum 
            if not node:
                return 0
            
            # Binary Tree 
            left = dfs(node.left)
            right = dfs(node.right)

            # Node.val은 겹칠 수 있으므로, object를 key로 사용 
            # NOTE: leaf노드에서 현재 노드까지의 path의 최대값, 왼쪽과 오른쪽 모두 음수인 경우, 현재 값과 더했을 때 더 작아지므로 0과 비교해준다. 
            dp[node] = max(left, right, 0) + node.val 
            # max_sum 후보 
            # 1. 현재 노드까지 path에서 가장 큰 값 
            max_sum = max(dp[node], max_sum)
            # 2. 왼쪽 -> 현재 노드 -> 오른쪽을 잇는 path와 현재 max_sum값 비교 
            max_sum = max(left+right+node.val , max_sum)
            return dp[node]
        
        dfs(root)

        return max_sum 

def build_tree(arr):
    if not arr:
        return None 
    
    root = TreeNode(arr[0])
    q = deque()
    q.append(root)
    i = 1 
    while q:
        node = q.popleft() 
        if i < len(arr) and arr[i]:
            node.left = TreeNode(arr[i])
            q.append(node.left)
        i += 1 
        if i < len(arr) and arr[i]:
            node.right = TreeNode(arr[i])
            q.append(node.right)
        i += 1 
    return root 

# arr = [-10,9,20,None, None,15,7] # 42 
# arr = [1, 2, 3] # 6 
# arr = [-10, -1, -3] # 6 
# arr = [2, -1] # 2
arr = [2, -1, -2] # 2
root = build_tree(arr)
sol = Solution()
print(sol.maxPathSum(root))

