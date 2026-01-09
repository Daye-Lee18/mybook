'''
the maximum sum of all "keys" of "any" sub-tree which is also a Binary Search Tree (BST)
- keys가 뭐지? = Node.val 
- 
Constraints 
- 1 <= N <= 4*1e4 
- -4*1e4 <= Node.val <= 4*1e4

예시:
root = [-4, -2, -5]
Output= 0 
-> 모든 값이 음수라서 빈 BST를 반환한다...
-> return 값은 0보다는 커야한다는 건가? 

Hint 
1. Create a datastructure with 4 parameters: (sum, isBST, mostLeft, mostRight).
    -> 예를 들어, [1, None, 10, -5, 20]을 보면, node 10에서는 BST이지만, root 1에서는 BST가 아닌 이유는, right subtree의 mostLeft값이 현재 노드 값인 1보다 작기 때문이다. 
    -> 즉, post-order로 tree를 돌때, 다음 아래의 2가지 조건을 만족해야, 현재 노드에서도 BST를 만족한다.  
        - right subtree의 mostLeft 값 > Node.val 
        - left subtree의 mostRight값 < Node.val
    -> 
2. In each node compute theses parameters, following the conditions of a Binary Search Tree.
'''

from typing import Optional 
# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

    
def build_tree(arr) -> Optional[TreeNode]:
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


MAX = int(1e9)

class NodeState:
    def __init__(self, isBST=False):
        self.sum = 0
        self.isBST = isBST
        self.mostLeft = MAX # 현재 노드의 left subtree중 가장 큰 값
        self.mostRight = -MAX # 현재 노드의 right subtree중 가장 작은 값 

class Solution:
    def maxSumBST(self, root: Optional[TreeNode]) -> int:

        maxsum = 0

        def postorder_dfs(node):
            nonlocal maxsum 

            if not node:
                return NodeState(True) # None node는 BST로 확정 
            
            leftState = postorder_dfs(node.left)
            rightState = postorder_dfs(node.right)

            curNodeState = NodeState() # leaf node부터는 isBST = False 기본값 
            
            if leftState.isBST and rightState.isBST:
                # 비교할때는 
                # left subtree의 가장 큰 값 < node.val < right subtree의 가장 작은 값 
                if leftState.mostRight < node.val < rightState.mostLeft: # 현재노드가 BST일 조건 
                    curNodeState.isBST= True 
                    curNodeState.sum = leftState.sum + rightState.sum + node.val 
                    # 현재 노드가 BST일때만, maxsum update 
                    maxsum = max(0, maxsum, curNodeState.sum)
            # 현재노드가 BST임과 상관없이, 이 값들은 계속 갱신됨
            # 갱신할때는, 다음 노드를 위해 현재 노드의 가장 왼쪽 값은 left 트리의 가장 작은 값이 되어야함. 
            curNodeState.mostLeft = leftState.mostLeft if leftState.mostLeft != MAX else node.val
            curNodeState.mostRight = rightState.mostRight if rightState.mostRight != -MAX else node.val

            return curNodeState
        
        postorder_dfs(root)
        return maxsum



# arr = [-4, -2, -5] # False, 0
# arr = [3, 1, 4, 0, 2, 1, 6] # 11 
# arr = [1, None, 3] # True, 4
# arr = [1, 2, None] # False, 2
# arr = [1, 2, 3] # False , 3
# arr = [2, 1, 3] # True, 6
# arr = [1, 4, 3, 2, 4, 2, 5, None, None, None, None, None, None, 4, 6 ] # False, 20 
# arr = [4, 3, None, 1, 2] # False, 2 
# arr = [5, 4, 8, 3, None, 6, 3] # 7
# arr = [1, None,10,-5,20] # 25
arr = [7,4,12,2,6,10,14,1,3,5,17,9,11,13,15] # 84 

root = build_tree(arr)
sol = Solution()
print(sol.maxSumBST(root))