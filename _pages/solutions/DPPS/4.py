'''
The diameter
- is the length of the "longest path" between any two nodes in a tree 
- the length of a path between two nodes is represented by the number of edges between them
- This path may or may not pass through the root

Idea
- 현재 노드의 height를 dp에 다 저장한다. 
- 현재 노드에서 최대 diameterd의 후보는 현재 노드의 양쪽 children의 높이를 더한 것이 된다.

Constarints 
- 1 <= the number of nodes <= 1e4 
- -100 <= Node.val <= 100 


-> 노드의 개수는 최대 1e4인데, node.val은 최대 201개이므로, 다른 위치에 있는 노드라도 중복된 값을 가질 수 있다. 
-> 따라서, visited를 설정할 때, node.val은 unique하지 않으므로 object별로 방문해는지 확인해야한다. 

Time Complexity:
- O(N): 모든 노드를 정확히 한 번 방문 
Space Complexity:
- O(H): 재귀 호출 스택, H=트리 높이, 최악의 경우 O(N)
'''

from typing import Optional 
from collections import deque 

# dp = defaultdict(int) # INIT to 0, node height (the # of edges)

# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
    
    def __repr__(self):
        return f"Node({self.val})"

class Solution:
    def diameterOfBinaryTree(self, root: Optional[TreeNode]) -> int:
        
        subtree_size = dict()
        max_diameter = 0

        def calculate_subtree_size(node):
            nonlocal subtree_size, max_diameter 
            if not node:
                return 0
            
            # 이거 없어도 DFS는 맨 아래서부터 위로 올라감. 
            # if node in subtree_size: 
            #     return subtree_size[node]

            
            left = calculate_subtree_size(node.left)
            right = calculate_subtree_size(node.right)
            
            subtree_size[node] = max(left, right) + 1 
            # node를 가운데로 했을 때의 지름 후보 left + right 
            max_diameter = max(max_diameter, left+right)
            return subtree_size[node]
        
        calculate_subtree_size(root)
        # print(subtree_size)
        return max_diameter

def build_tree(arr):
    idx = 0 
    root = TreeNode(arr[idx])
    idx += 1 

    q = deque()
    q.append(root)
    while q and idx < len(arr):
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

def tree_to_arr(root:TreeNode):
    if not root:
        return []
    
    result = []
    q = deque()
    q.append(root)

    while q:
        node = q.popleft()

        if node:
            result.append(node)
        else:
            result.append(None)
            continue 
        
        q.append(node.left)
        q.append(node.right)

    while result and result[-1] == None:
        result.pop()
    return result 

    
# tree = [1,None, 2,None, None, 3,4] 
# tree = [1, 2, 3, 4, 5] # 3 
# tree = [1, 2] # 1 
tree = [1, 2, 3, None, None, 4, 5, 6, None, None, 7, 8, None, None, 9] # 6
root = build_tree(tree)
# print(tree_to_arr(root))
sol = Solution()
print(sol.diameterOfBinaryTree(root))