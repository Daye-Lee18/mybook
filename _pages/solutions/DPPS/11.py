from typing import List 
'''
Idea:
- binary lifing (a.k.a. doubling / sparse table): store only ancestors at distances 2*j
    - Use sparse table (dp application) to tralvel the tree upwards in a fast way
- memory: O(nlogn)
- time: O(nlogn)

Constraints:
- 1 <= k, n <= 5*1e4 
- there will be at most 5*1e4 queries ~ O(5*1e4)

NOTE: k-ary tree (not binary)
'''

class TreeAncestor:
    def __init__(self, n: int, parent: List[int]):
        '''
        parent[i] = the parent of `ith` node 
        the root node = 0 
        '''
        self.n = n
        self.LOG = (self.n).bit_length()
        self.up = [[-1]*n for _ in range(self.LOG)] # self.up[j][v] = 2^j-th ancestor of v, or -1
        self.up[0] = parent[:] # 2^0 = 1st parent 

        for j in range(1, self.LOG):
            prev = self.up[j-1]
            cur = self.up[j]
            for v in range(n):
                p = prev[v]
                cur[v] = -1 if p == -1 else prev[p] 




    # Time Complexity : O(1)
    def getKthAncestor(self, node: int, k: int) -> int: 
        '''
        Find the `kth` ancestor of a given node 
        -> `kth` ancestor of a tree nodei s the `kth` node in the path that node to the root node
        -> if there is no such ancestor return -1  
        -> 1 <= k <= 5*1e4 
        '''
        # jump using bits of k 
        j = 0
        while node != -1 and k:
            if k&1:
                node = self.up[j][node]
            k >>= 1 # at most 16 jumps 
            j += 1 
        return node 
        
        

# n = 7; parent = [-1, 0, 0, 1, 1, 2 ,2]; node = 3; k=1 # 1 
n = 7; parent = [-1, 0, 0, 1, 1, 2 ,2]; node = 5; k=2 # 0
# n = 7; parent = [-1, 0, 0, 1, 1, 2 ,2]; node = 6; k=3 # -1
# Your TreeAncestor object will be instantiated and called as such:
obj = TreeAncestor(n, parent)
param_1 = obj.getKthAncestor(node,k)
print(param_1)