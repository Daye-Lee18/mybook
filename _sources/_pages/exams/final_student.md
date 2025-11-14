# Final

## Short answer problems 

- For each question below, fill in the blanks labeled (1), (2), (3), …, or provide a brief answer as requested.
- Record all short-answer responses on your own answer sheet, clearly numbered to match the questions.
- Each correctly completed blank (e.g., (1), (2)) is worth 2 points.

### Q1. Insertion sort and selection sort

Insertion sort is a sorting algorithm that places each element into its appropriate (1) among already sorted elements on its left. It takes (2) time complexity on average and in the worst case. On the other hand, selection sort(in ascending order) selects the (3) element from the unsorted part and (4) it with the current position. And selection sort algorithm takes (5) time complexity. Insertion sort assumes that (6), so sorting starts from the second element, not the first element.

```{admonition} A1
:class: dropdown

A1. <br>
(1) index  <br>
(2) O($N^2$)  <br>
(3) smallest  <br>
(4) swap <br>
(5) O($N^2$) <br>
(6) the data ahead of the current data is already sorted <br>
```

### Q2. Quick sort 

In the Quick Sort Algorithm, the concept of (1) is used, where the array is divided into two subarrays based on a pivot. This process should proceed until (2). According to our lecture, we use two pointers (left and right) along with the pivot index to perform (3). In (3), a pivot is the first element in a list and we find the larger number than the pivot from left and the smaller number than the pivot from right and swap them unless their position is not located across. If implemented with inclusive `end` index, the recursive termination condition is (4). The average time complexity is (5). 

```{admonition} A2
:class: dropdown

A2.  <br>
(1) partitioning <br>
(2) The subarray size becomes 1 <br>
(3) Hoare Partitioning <br>
(4) start >= end  <br>
(5) O(NlogN)
```

### Q3. Count Sort
Count sort can be used under certain conditions. <br>
First, list all conditions. <br>
Second, provide ***at least two*** real-world examples where count sort is applicable.

1. Conditions: <br>
2. Examples where count sort can be used: <br>

```{admonition} A3
:class: dropdown 

A3.  <br>
- Conditions:
    - The range of values (max − min) must be relatively small.
    - The data values must be integers (typically non-negative).

- Examples:
    - Sorting students’ exam scores (e.g., 0–100).
    - Sorting ages in a classroom (e.g., 6–12 years old).
    - Sorting shoe sizes in a store (typically small integer range).
```

### Q4. Sorting library in python 

There is a built-in function in Python, (1), which returns a new list and does not modify the original data (not in-place). It takes an iterable data type as an argument. Also, for a list data type, it has a built-in method - (2) sorting the list in-place. Both functions arrange the data in (3) order by default. 

And in the class, we learned about `key` parameter and it takes a (4). It comes in handy when the elements in the array are an iterable data type such as tuple, list, etc. 

```{admonition} A4
:class: dropdown 

A4. <br>
(1) sorted() <br>
(2) list.sort() <br>
(3) ascending <br>
(4) function <br>
```


### Q5. List slicing 1

Based on the same examples above, rewrites each slicing expression by explicitly filling in the omitted values for  `start`, `end`, `step`. <br>

For example, the slice `arr[::1]` becomes `arr[0:len(arr):1]`. For n=10, this would be `arr[0:10:1]`. 

```python
arr = [1,2,3,4,5,6,7,8,9,10]

# 1)
print(arr[:3])
# 2) 
print(arr[-3:])
# 3)
print(arr[:-3])
# 4) 
print(arr[:-3:-1])
# 5)
print(arr[:-3:-2])
# 6)
print(arr[:-5:-2])
# 7)
print(arr[:3:-1])
# 8)
print(arr[:3:1])
# 9)
print(arr[::-1])
# 10)
print(arr[::-2])
# 11)
print(arr[5::-1])
# 12)
print(arr[5::-2])
# 13)
print(arr[-100::5])
# 14)
print(arr[::-1][::2])
# 15)
print(arr[::2][::-1])
```

```{admonition} A5
:class: dropdown 

(1) arr[0:3:1] <br>
(2) arr[-3:10:1] == arr[7:10:1] <br> 
(3) arr[0:-3:1] == arr[0:7:1] <br>
(4) arr[9:-3:-1] <br>
(5) arr[9:-3:-2]  <br>
(6) arr[9:-5:-2] <br>
(7) arr[9:3:-1] <br>
(8) arr[0:3:1] <br>
(9) arr[9:-1:-1]<br>
(10) arr[9:-1:-2] <br>
(11) arr[5:-1:-1] <br>
(12) arr[5:-1:-2] <br>
(13) arr[-100:10:5] == arr[0:10:5] <br>
(14) arr[9:-1:-1][0:10:2] <br>
(15) arr[0:10:2][4:-1:-1] == arr[0:10:2][4::-1] <br>
```

### Q6. List slicing 2 

```python
print(arr[::0]) 
```

In the above code, can this code run? If not, why not?

```{admonition} A6
:class: dropdown 

No, it cannot run. In Python slicing, the step value cannot be 0.
If `step` = 0, Python cannot determine how to move through the list, so it raises a ValueError.
```

### Q7. Quick Sort Implementation 

The following source code is a code for the quick sort. Fill in the blanks (blank(1), blank(2), blank(3), etc).

```python
import random

N = 15
my_list = [random.randint(1, 100) for i in range(N)]

def quick_sort(arr, start, end): # end pointer: inclusive 
    if start blank(1) end:
        return 
    
    pivot = start 
    left = start + 1 
    right = end # inclusive 
    # partitioning 
    while left <= right:
        while left blank(2) end and arr[left] <= arr[pivot]:
            left += 1 
        while right blank(3) start and arr[right] >= arr[pivot]:
            right -= 1 

        if left <= right:
            arr[left], arr[right] = arr[right], arr[left]
            continue 
        else:
            blank(4)
            break # partitioning done 
    
    # pivot element is at right index 
    quick_sort(arr, blank(5), right-1)
    quick_sort(arr, right+1, blank(6))
        
print(f"before sorting: {my_list}")
quick_sort(my_list, 0, N-1)
print(f"after sorting: {my_list}")
```


```{admonition} A7
:class: dropdown 

blank(1) <= <br>
blank(2) <= <br>
blank(3) > <br>
blank(4) arr[pivot], arr[right] = arr[right], arr[pivot] <br>
blank(5) start <br>
blank(6) end <br>
```

### Q8. Tree 

Tree is a graph without a (1). There are several properties of a tree. State True if the following sentence(s) is correct, State False the sentence(s) is wrong. 

(2) There is always a unique path between any two nodes in a tree. 
(3) A tree must have a root .
(4) Removing any one edge in a tree splits it into two smaller trees (subtrees).
(5) A child node cannot have children. 
(6) A tree does not inherently need a root, but if we choose one, we can view the structure as a rooted tree. 
(7) In a rooted tree, for any node, all adjacent nodes except its parent are its children.

```{admonition} A8
:class: dropdown 

(1) cycle <br>
(2) True <br>
(3) False <br>
(4) True <br>
(5) False <br>
(6) True <br>
(7) True <br> 
```


### Q9. Edges to parent, children list

Using the property of a tree that all adjacent nodes of a node are its children except for one node (its parent), we can construct the tree in a list form. The following source code defines a function `edges_to_parent_children_list` which takes the list of `edges` and `N` (the number of nodes), and returns the parent and children lists. Note that the nodes are 1-indexed (starting from 1).

Fill in the blanks in the following code. 

```python
'''
edges to children, parents list 
'''

from typing import List, Tuple 
from collections import deque 

def edges_to_parent_children_list(N:int, edges: List[Tuple[int, int]], root=1) -> Tuple[List, List]:
    parent = [-1] * (N+1) # 1-indexed 
    children = [[] for _ in range(N+1)]
    graph = [[] for _ in range(N+1)]

    for edge in edges:
        graph[edge[0]].append(edge[1])
        graph[edge[1]].append(edge[0])

    q = deque([root])
    parent[root] = 0  # root has no parent 
    while q:
        cur_val = q.popleft()

        for nxt_node in graph[cur_val]:
            if parent[nxt_node] != -1: # already visited 
                continue 
            blank(1)
            blank(2)
            blank(3)
    return parent, children 


n = 7
edges = [(1,2),(1,3),(2,4),(2,5),(3,6),(3,7)]
root = 1

p, ch = edges_to_parent_children_list(n, edges)
print("parent:", p)       # [ -1, 0, 1, 1, 2, 2, 3, 3 ]
print("children:", ch)    # [ [], [2,3], [4,5], [6,7], [], [], [], [] ]
```
````{admonition} A9
:class: dropdown 

```{code-block} python
parent[nxt_node] = cur_val 
children[cur_val].append(nxt_node)
q.append(nxt_node)
```
````

### Q10. BST 

BST is a short term for (1). It is a tree data structure designed for efficient searching. The root value must be (2) than every value in the left subtree and the (3) than every value in the right subtree. An interesting characteristic of the BST is that when you traverse it using (4), the resulting sequence is in ascending order. The time complexity for this traversal for N nodes is (5). 

````{admonition} A10
:class: dropdown 

(1) Binary Search Tree <br>
(2) greater <br>
(3) smaller <br>
(4) in-order DFS traversal <br> 
(5) O(N), In-order traversal visited every node exactly once. <br> 
````

### Q11. Graph Algorithm 

A graph consists of vertices and edges. In particular, we learned about (1) data structure, which maintains multiple (2) such that no two sets share any common elements. Two core operations of this data structure are (3) and (4): the former finds the representative (or root) of the set, and the latter merges two sets into one.

````{admonition} A11
:class: dropdown 

(1) Union-Find (Disjoint Set Union, DSU) <br>
(2) Disjoint sets <br>
(3) Find <br>
(4) Union<br>
````

### Q12. Kruskal's Algorithm  

Kruskal's algorithm is an algorithm used to find out (1) in a graph. It is a (2) algorithm because it always selects the locally optimal choice (the smallest available edge) at each step. We learned about the algorithm in the class. Describe the algorithm steps below. (3) 

````{admonition} A12
:class: dropdown 

(1) MST (Minimum Spanning Tree) <br>
(2) Greedy <br>
(3) 
3-1. Sort all edges in non-decreasing (ascending) order of their weights. <br>
3-2. Consider edges one by one from smallest to largest.<br>
3-3. For each edge, use Union-Find to check whether adding the edge forms a cycle.<br>
3-4. If it does not form a cycle, include the edge in the MST.<br>
3-5. Continue until (N - 1) edges have been added to the MST. <br>
````

### Q13. Topological Sort 

Topological sort is an ordering of the vertices in a directed graph. It can only be applied to a (1) which is a directed graph with no cycles. (2) is one of the methods to perform topological sorting using the concept of indegree. Indegree is defined as (3). A node with indegree 0 can be considered as (4) in the resulting sorted order. Topological sorting can also be used to detect whether a graph contains a cycle. If during the process (5), then we can conclude that there exists a cycle in the graph. The time complexity of this algorithm is (6), where V is the number of vertices and E is the number of edges.

Give at least three real-life examples where topological sorting can be applied. (7)


````{admonition} A13
:class: dropdown 

(1) Directed Acyclic Graph (DAG) <br>
(2) Kahn's Algorithm <br>
(3) the number of incoming edges to the node <br> 
(4) starting node / a valid first element in the ordering (a node that can be placed first) <br>
(5) there are still nodes not processed but no node with indegree 0 exists <br>
(6) O(V+E) <br>
(7) 
7-1. Course prerequisites scheduling <br>
7-2. Build/compile order of software modules <br>
7-3. Task scheduling in project planning (dependency ordering) <br>
````

### Q14. Topological Sort Code 

Fill in the blanks (blank(1), blank(2), ...etc) to make the code work. 

```python 
from typing import List, Tuple
from collections import deque 

def topological_sort(N: int, edges:List[Tuple[int, int]]):
    '''
    N: the number of nodes 
    edges: edges in the graph 
    Note that the graph is 1-indexed. 
    '''

    indegree = [0] * (N+1)
    graph = [[] for _ in range(N+1)]

    for edge in edges: # edge[0]: starting point, edge[1]: end point 
        graph[edge[0]].append(edge[1])
        indegree[blank(1)] += 1 

    q = deque([node for node in range(1, N+1) if blank(2)])
    result = []
    while q:
        cur_node = q.popleft()
        result.append(cur_node)

        for nxt_node in graph[cur_node]:
            indegree[nxt_node] -= 1
            if blank(3):
                q.append(nxt_node)

    return blank(4)
```
````{admonition} A14
:class: dropdown 

blank(1): edge[1] <br>
blank(2): indegree[node] == 0 <br>
blank(3): indegree[nxt_node] == 0 <br>
blank(4): result <br>

````

### Q15. Prefix Sum 

A prefix refers to the segment of data from the beginning (index 0) up to a specific position. A prefix sum is the cumulative sum of elements from the start to that position. The reason for constructing a prefix sum array  is to calculate (1) in (2) time complexity. The prefix sum array is typically of size N+1 when the original array has size N, and prefix[0] is set to (3). In the lecture, we learned about (4) to make it easier to compute (1) with 1-indexed problems. 

````{admonition} A15
:class: dropdown 

(1) a range sum <br>
(2) O(1) <br>
(3) 0 <br>
(4) zero padding <br>
````

### Q16. Prefix sum code 

The following source code is for a range sum of 1D and 2D data array. Fill in the blanks. 

```python
from typing import List 

def build_1D_prefix_array(data:List[int]) -> List[int]:
    """
    Build a 1D prefix-sum array with zero padding.
    px has length n+1; px[0] = 0 and px[i] = sum(data[:i]).
    """
    n = len(data)
    px = [0]*(n+1)

    for idx in range(1, n+1):
        px[idx] = blank(1)
    
    return px 

def calculate_range_sum_1D(px:List[int], start:int, end:int) -> int:
    """
    Return sum of data[start..end] (inclusive), assuming px is zero-padded.
    """
    return blank(2)


def build_2D_prefix_array(data:List[List[int]]) -> List[List[int]]:
    """
    Build a 2D prefix-sum (summed area) array with zero padding.
    px has shape (H+1) x (W+1); px[y][x] is sum over data[0..y-1][0..x-1].
    """
    H = len(data); W = len(data[0])
    px = [[0]*(blank(4)) for _ in range(blank(5))]

    for y in range(1, H+1):
        for x in range(1, W+1):
            px[y][x] = blank(3)
    return px 

def calculate_range_sum_2D(px:List[List[int]], y1, x1, y2, x2) -> int:
    """
    Return sum over inclusive rectangle (y1, x1) .. (y2, x2).
    Assumes 1-based coordinates relative to data, and px is zero-padded.
    """
    return blank(6)
```

````{admonition} A16
:class: dropdown 

blank(1): px[idx-1] + data[idx-1] <br>
blank(2): px[end+1]-px[start] <br>
blank(3): data[y-1][x-1] + px[y-1][x] + px[y][x-1] - px[y-1][x-1] <br>
blank(4): W + 1 <br>
blank(5): H + 1 <br>
blank(6): px[y2][x2] - px[y2][x1-1] - px[y1-1][x2] + px[y1-1][x1-1] <br>
````

### Q17. XOR Properties 

XOR (Exclusive OR) is a bitwise operation. When applying XOR, it returns (1) when the two bits are the same, and returns (2) when they are different. The operator in Python is (3). This operation appears in many problems, such as toggling bits, (4), and simple cipher creation. Its identity element is (5), and (3) applied to the same value satisfies: a (3) a = (6).

````{admonition} A17
:class: dropdown 

(1) 0 <br>
(2) 1 <br>  
(3) ^ <br>
(4) finding a unique element in the array <br>
(5) 0 <br>
(6) 0 <br> 
````

### Q18. XOR calculation 

Calculate the following mini problems and give each answer in ***decimal form***

***Note***: A decimal number is the regular base-10 number system we use in everyday arithmetic.

(1) 3 ^ 3 ^ 8 ^ 1 <br>
(2) 4 ^ 4 <br>
(3) 5 ^ 8 <br>
(4) 10 ^ 12 <br>
(5) 5 ^ 6 ^ 7 ^ 6 ^ 8<br>
(6) 7 ^ 2 ^ 3<br>
(7) 9 ^ 36 <br>

````{admonition} A18
:class: dropdown 

(1) 9 <br>
(2) 0 <br>
(3) 13 <br>
(4) 6 <br>
(5) 10 <br>
(6) 6 <br>
(7) 45 <br>

````

### Q19. Prefix XOR Source code 

Fill in the blanks (blank(1), blank(2), etc.) in the following code. 

```python
from typing import List 

def build_prefix_XOR(data:List[List[int]]) -> List[List[int]]:
    """
    Build 2D prefix XOR with zero padding.
    px has shape (H+1) x (W+1).
    px[y][x] = XOR over data[0..y-1][0..x-1].
    """
    H = len(data); W = len(data[0])
    px = [[0] * blank(1) for _ in range(blank(2))]

    for y in range(1, H+1):
        for x in range(1, W+1):
            px[y][x] = blank(3)
    return px 


def range_XOR(px:List[List[int]], y1: int, x1: int, y2: int, x2: int) -> int:
    """
    Return XOR over inclusive rectangle (y1, x1) .. (y2, x2),
    assuming (y1, x1, y2, x2) are 1-based indices on the original data.
    """
    return blank(4)
```

````{admonition} A19
:class: dropdown 

blank(1): W+1 <br>
blank(2): H+1 <br>
blank(3): data[y-1][x-1] ^ px[y-1][x] ^ px[y][x-1] ^ px[y-1][x-1] <br>
blank(4): px[y2][x2] ^ px[y2][x1-1] ^ px[y1-1][x2] ^ px[y1-1][x1-1] <br>
````

### Q20. Numpy 

```python
import numpy as np 

data = np.arange(blank(1), blank(2)).reshape(blank(3))

print(data)
'''
output: 
[[ 1  2  3  4  5  6]
 [ 7  8  9 10 11 12]
 [13 14 15 16 17 18]
 [19 20 21 22 23 24]]
'''
```

````{admonition} A20
:class: dropdown 

blank(1): 1<br>
blank(2): 25<br>
blank(3): (4, 6) <br>
````

### Q21. Bisect_left vs. Binary Search 

The `bisect_left()` function from the bisect module returns the index where a given value should be inserted in order to keep the list sorted. It returns the leftmost position where the value can fit. In other words, it finds the smallest index i such that arr[i] >= value.

The logic of `bisect_left()` is similar to that of binary search. Fill in the blanks in the following code. <br>

Then, write the expected output of the provided code.

Note: print() uses end="\n" by default.


```python
from typing import List 

def bisect_left(arr: List[int], value: int) -> int:
    left = 0; right = len(arr) # exclusive 

    while left blank(1) right:
        mid = blank(2)

        if arr[mid] blank(3) value:
            left = mid + 1 
        else:
            right = blank(4)
    
    return left 

def binary_search(arr: List[int], target:int) -> int:
    left = 0; right = len(arr) # exclusive 

    while left blank(1) right:
        mid = blank(2)

        if arr[mid] == target:
            return mid 
        
        if arr[mid] blank(3) target:
            left = mid + 1 
        else:
            right = blank(4)
    
    return blank(5)

arr = [1, 2, 3, 3, 3, 4, 5, 9, 12]
target= 6

print(bisect_left(arr, target))
print(binary_search(arr, target))
```

````{admonition} A21
:class: dropdown 

blank(1): < <br>
blank(2): (left + right) // 2  <br>
blank(3): < <br>
blank(4): mid <br>
blank(5): -1 <br>

Expected results:<br>
7 <br>
-1 <br>
````



## Problem solving problems 

### Q1. Build Tree 

Implement `build_tree(arr: List) -> Optional[Node]` and `root_to_list(root:Node) -> List[int]` in the following source code. 

```python
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


def root_to_list(root:Node) -> List[int]:


if __name__ == "__main__":
    arr = [1, 2, 3, None, 5, None, 7]
    root = build_tree(arr)
    print(root_to_list(root))
```
````{admonition} solution
:class: dropdown 

```{code-block} python
from collections import deque 
from typing import List, Optional 

class Node:
    def __init__(self, val:int, left=None, right=None):
        self.val = val 
        self.left = left 
        self.right= right 

    def __repr__(self):
        return f"TreeNode{self.val}"
    
def build_tree(arr) -> Optional[Node]:
    if len(arr) == 0:
        return None 
    
    root = Node(arr[0])
    idx = 1 

    q = deque([root])
    while idx < len(arr):
        cur_node = q.popleft()

        if idx < len(arr) and arr[idx]:
            cur_node.left = Node(arr[idx])
            q.append(cur_node.left)
        idx += 1 
        
        if idx < len(arr) and arr[idx]:
            cur_node.right = Node(arr[idx])
            q.append(cur_node.right)
        idx += 1 
    return root 

def root_to_list(root:Node) -> List[int]:
    if not root:
        return []
    
    q = deque([root])
    res = []
    while q:
        cur_node = q.popleft()

        # res.append(cur_node)
        if not cur_node:
            res.append(None)
            continue 
        res.append(cur_node.val)

        q.append(cur_node.left)
        q.append(cur_node.right)
    
    # trim trailing 
    while res and not res[-1]:
        res.pop()
    return res 

if __name__ == "__main__":
    arr = [1, 2, 3, None, 5, None, 7]
    root = build_tree(arr)
    print(root_to_list(root))
```
````
### Q2. Sorting 

Given the array `nums` of size `n`, return the majority element. 

The majority element is the element that appears more than $\lfloor n/ 2 \rfloor$ times. You may assume that the majority element always exists in the array. 

Example 1: 
- Input: nums = [3, 2, 3]
- Output: 3 

Example 2: 
- Input: nums = [2, 2, 1, 1, 1, 2, 2]
- Output: 2 

Constraints:
- n == nums.length
- 1 <= n <= 5 * $10^4$
- $-10^9$ <= nums[i] <= $10^9$
- The input is generated such that a majority element will exist in the array 

````{admonition} Solution
:class: dropdown 

```python
from typing import List 

class Solution:
    def majorityElement(self, nums: List[int]) -> int:
        nums.sort()

        cnt = 0 
        majority_num = nums[0]
        for num in nums:
            if majority_num == num:
                cnt += 1 
            else:
                cnt -= 1 
                if cnt == 0:
                    majority_num = num

        return majority_num
    
# nums = [3, 2, 3]
nums = [2,2,1,1,1,2,2]
sol = Solution()
print(sol.majorityElement(nums))
```
````