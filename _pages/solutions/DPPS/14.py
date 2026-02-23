from typing import List 

'''
k-ary tree 
0-indexed n nodes tree 
- parent[i] = parent of ith node 
- root = node 0, parent[0] = -1 

(1) lock 
- locks the given node for the given user and prevents other users from locking the same node. 
    - 현재 사용자가 locking하면, 다른 사용자는 해당 노드를 Locking하지 못함. 
- You may only lock a node using this function if the node is unlocked 

(2) unlock 
- Unlocks the given node for the given user. 
- You may only unlock a node using this function if it is currently locked by "the same user" (잠금 사용자가 동일해야만, unlock가능)

(3) upgrade 
- locks the given node for the given user 
    - self.lock() function 사용 
- unlocks all of its descendants regardless of who locked it. (잠금 사용자가 누구이든지 상관없이 unlock 가능)
    - NOTE: 해당 노드의 descendants nodes들도 다 unlock해야함. 
- You may upgrade a node if all 3 conditions are true: 
    - 1. The node is unlocked, 
    - 2. It has at least one descendant (by any user), and 
    - 3. It does not have any locked ancestors 
        - 현재 잠그려는 node의 ancestors들 중 하나라도 locking되어 있으면 안됨. 
        - NOTE: ancestors를 traverse하는 function 

        
Constraints 
- n == parent.length 
- 2 <= n <= 2000
- 0 <= num <= n-1 
- 1 <= user <= 10^4 
- at most 2000 calls in total will be made to lock, unlock, and upgrade 
- Tree depth at most 11 
'''
class LockingTree:

    def __init__(self, parent: List[int]):
        self.parent = parent
        self.direct_children = [[] for _ in parent]
        for i, x in enumerate(parent): 
            if x != -1: self.direct_children[x].append(i)
        self.locked = {}

    def lock(self, num: int, user: int) -> bool:
        if num in self.locked: return False 
        self.locked[num] = user
        return True 

    def unlock(self, num: int, user: int) -> bool:
        if self.locked.get(num) != user: return False 
        self.locked.pop(num)
        return True 

    def upgrade(self, num: int, user: int) -> bool:
        if num in self.locked: return False # check for unlocked
        
        node = num
        while node != -1:  # at most ~O(11) 
            if node in self.locked: break # locked ancestor
            node = self.parent[node]
        else: 
            stack = [num]
            locked_descendant = []
            # locked_descendant에 현재 locked 되어있는 descendant 넣기 
            while stack: 
                node = stack.pop()
                if node in self.locked: locked_descendant.append(node)
                for child in self.direct_children[node]: stack.append(child)
            # locked_descendant를 unlock해주기 
            if locked_descendant: 
                self.locked[num] = user # lock given node 
                for node in locked_descendant: self.locked.pop(node) # O(1), unlock all descendants
                return True 
        return False # locked ancestor 


# Your LockingTree object will be instantiated and called as such:
# obj = LockingTree(parent)
# param_1 = obj.lock(num,user)
# param_2 = obj.unlock(num,user)
# param_3 = obj.upgrade(num,user)
import math 

n = 2000 
print(math.log2(2000)) # ~ 11 
print(n.bit_length()) # ~ 11 