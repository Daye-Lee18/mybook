from typing import Optional 

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    def minCameraCover(self, root: Optional[TreeNode]) -> int:

        if not root: return 0

        self.cameras = 0 

        '''
        dfs function returns the number of nodes covered and 
        `self.cameras` keeps track on the overall camera 
        '''
        def dfs(node: Optional[TreeNode]) -> int:
            if not node:
                return 0
            
            if (not node.left) and (not node.right):
                return 1 
            
            left = 3 ;right= 3  # default value 
            if (node.left):
                left = dfs(node.left)
            if (node.right):
                right = dfs(node.right)

            if left == 1 and right == 1:
                self.cameras += 1 
                return 2 
            
            if left == 3 and right == 3:
                return 1 
            
            if left >= 2 and right >= 2:
                return 3 
        
        ## run 
        cover = dfs(root)
        if cover == 1:
            self.cameras += 1 
        return self.cameras 

