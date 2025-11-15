from collections import deque 
from typing import List, Optional 

class Node:
    def __init__(self, val:int, left=None, right=None):
        self.val = val 
        self.left = left 
        self.right= right 

    def __repr__(self):
        return f"TreeNode{self.val}"
    
   
def build_tree(arr: List) -> Optional[Node]:
    root = arr[0]
    q = deque[(root)]

    while q:
        cur_node = q.popleft()
        for i in range(1, len(arr)):
            if i % 2 == 0:
                cur_node.left = arr[i]
                q.append(arr[i])
                continue
            else:
                cur_node.right = arr[i]
                q.append(arr[i])

def root_to_list(root:Node) -> List[Node]:
    q = deque[(root)]
    list = []
    while q:
        cur_node = q.popleft()
        list.append(cur_node.val)
        q.append(cur_node.left)
        q.append(cur_node.right)
    newList = []
    for i in range(len(list)):
        if i != None:
            newList.append(i)
    return newList

def run_test():
    test_case = [
        # [[], []],
        [[1], [1]],
        [[1, 2, 3], [1, 2, 3]],
        [[1, None, 2, 3],[1, None, 2, 3]],
        [[1, 2, None, 3, 4],[1, 2, None, 3, 4]],
        [[1, 2, 3, None, 4, None, 5],[1, 2, 3, None, 4, None, 5]],
        [[1, None, 2, None, 3, None, 4],[1, None, 2, None, 3, None, 4]],
        [[1, 2, 3, 4, 5, 6, 7, 8],[1, 2, 3, 4, 5, 6, 7, 8]],
        [[1, 2, 3, None, None, 6, 7],[1, 2, 3, None, None, 6, 7]],
        [[1, None, None], [1]],
        [[0], [0]], # 11 
        [[1, 0, 2, 0,3], [1, 0, 2, 0, 3]],
        [[1, 2, 0, 0, 3, 0, 4], [1, 2, 0, 0, 3, 0, 4]], # 13 
        [[1, 2, 3, None, None, None, 4, None, 5], [1, 2, 3, None, None, None, 4, None, 5]],
        [[1, 2, 3, 4, None, None, None, 5], [1, 2, 3, 4, None, None, None, 5]],
        [[1, 2, 3, 4, 5, None, None, None, None, 6], [1, 2, 3, 4, 5, None, None, None, None, 6]],
        [[1, 2, 3, 4, None, 6, None, 7, None, None, None], [1, 2, 3, 4, None, 6, None, 7]],
        [[1, 2, 3, None, 4, 5, None, None, None, 6], [1, 2, 3, None, 4, 5, None, None, None, 6]],
        [[1, 2, 3, 4, 5, 6, None, None, None, None, 7], [1, 2, 3, 4, 5, 6, None, None, None, None, 7]],
        [[0, 1, 0], [0, 1, 0]]
    ]


    nums = len(test_case)
    passed = 0
    for idx in range(nums):
        if idx == 10:
            print('a')
        output = root_to_list(build_tree(test_case[idx][0]))
        
        ok = output == test_case[idx][1]
        print(f"Test case {idx+1}: Input: {test_case[idx][0]}, Output: {output} (Expected:{test_case[idx][1]} -> {'OK' if ok else 'FAIL'})")
        print()
        passed += ok
    
    return passed, nums



if __name__ == "__main__":
    passed, total_case= run_test()
    print(f"Passed {passed}/{total_case}")