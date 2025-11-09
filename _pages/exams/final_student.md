# Final

## Short answer problems 

- For each question below, fill in the blanks labeled (1), (2), (3), …, or briefly describe what is being asked.
- Write all short-answer responses in a file named `answer.md` and include it in your submission.
- Each correctly answered blank (e.g., (1), (2)) is worth 2 points.

### Q1. Insertion sort and selection sort

Insertion sort is an sorting algorithm that attempt to find a propriate (1) of each data, taking (2) time complexity. On the other hand, selection sort, in ascending order, is another type of sorting algorithm, arranging data by pick the (3) data at each step and swap. And selection sort alogrithm takes (4) time complexity. Selection sort assumes that (5), so sorting the data starts from the second element not the first element in the array. 

참고) When writing in a .md file, to express exponents such as $2^3$, use the caret `^` symbol inside math mode, e.g., write `$2^3$`. Alternatively, you may write 2**3 for simplicity.

```{admonition} A1
:class: dropdown

A1. <br>
(1) index  <br>
(2) O($N^2$)  <br>
(3) smallest  <br>
(4) O($N^2$) <br>
(5) the data ahead of the current data is already sorted <br>
```

### Q2. Quick sort 
In Quick sort algorithm, the concept of (1) comes in, where we divide the array into two array pivoting the current element. This process should proceed until (2). To implement, according to our lecture, we use (3) pointers, start, end, and mid pointer. Specifically, we learned about (4). In (4), we determine pivot as the first element in a list and we find the larger number than the pivot from left and the smaller number than the pivot from right and swap them unless their position is not located across. If we implement the code using DFS and the end pointer is inclusive, the termination code of the recursive function should be (5), indicating that there is only one element is left in the list. The time complexity for this algorithm is (6) in average. 

```{admonition} A2
:class: dropdown

A2.  <br>
(1) partioning <br>
(2) there is only one element left in the current array.  <br>
(3) 3 <br>
(4) Hoare Partitioning 
(5) start >= end  <br>
(6) O(NlogN)
```

### Q3. Count Sort
Count sort can be applied under certain conditions. First, list all condiitons and Second, give ***at least two*** life-related examples where you can use in the real-world. 

1. conditions: <br>
2. examples where count sort can be used: <br>

```{admonition} A3
:class: dropdown 

A3.  <br>
- conditions: 
    - the biggest and the smallest data should not be too big 
    - number should be natural numbers (non-negative)
- examples: 
    - students' midterm scores sorting 
    - prices of all products in a toy store 
```

### Q4. Sorting library in python 
There is a built-in library in Python which is (1) where the data is sorted and take not in-place, meaning you should store it somewhere not to lost the rearranged data. It takes an iterable data type as an argument. Also, in a list data type, it has a built-in method - (2), the sorting takes place in-place. Both functions arrange the data in (3) order. 

And in the class, we learned about `key` parameter and it takes a (4). It comes handy when the elements in the array is an iterable data type such as tuple, list, etc. 

```{admonition} A4
:class: dropdown 

A4. <br>
(1) sorted() <br>
(2) list.sort() <br>
(3) ascending <br>
(4) function <br>
```


### Q5. List slicing 1

Based on the same examples above, give an answer for each code line that including ommited value for  `start`, `end`, `step`. For example, for the question of `arr[::1]`, the answer is `arr[0:len(arr):1]`. for n=10, the final answer would be `arr[0:10:1]`. 

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
(2) arr[-3:10:1] <br> 
(3) arr[0:-3:1]<br>
(4) arr[9:-3:-1] == arr[9:7:-1] <br>
(5) arr[9:-3:-2] == arr[9:7:-2] <br>
(6) arr[9:-5:-2] == arr[9:5:-2] <br>
(7) arr[9:-3:-1] == arr[9:7:-1] <br>
(8) arr[0:3:1] <br>
(9) arr[9:-1:-1]<br>
(10) arr[9:-1:-2] <br>
(11) arr[5:-1:-1] <br>
(12) arr[5:-1:-2] <br>
(13) arr[-100:10:5] == arr[0:10:5] <br>
(14) arr[9:-1:-1][0:10:2] <br>
(15) arr[0:10:2][9:-1:-1] <br>
```

### Q6. List slicing 2 

```python
print(arr[::0]) 
```

In the above code, can the code be running? If not, why?


```{admonition} A6
:class: dropdown 

It won't be running since the step is either bigger than 0 and less than 0, can't be equal to 0. 
```

### Q7. Quick Sort Implementation 

The following source code is a code for the quick sort. Fill in the blank ((1), (2), (3), etc).

```python
import random

N = 15
my_list = [random.randint(1, 100) for i in range(N)]

def quick_sort(arr, start, end): # end pointer: inclusive 
    if start (1) end:
        return 
    
    pivot = start 
    left = start + 1 
    right = end # inclusive 
    # partitioning 
    while left <= right:
        while left (2) end and arr[left] <= arr[pivot]:
            left += 1 
        while right (3) start and arr[right] >= arr[pivot]:
            right -= 1 

        if left <= right:
            arr[left], arr[right] = arr[right], arr[left]
            continue 
        else:
            (4)
            break # partitioning done 
    
    # pivot element is at right index 
    quick_sort(arr, (5), right-1)
    quick_sort(arr, right+1, (6))
        
print(f"before sorting: {my_list}")
quick_sort(my_list, 0, N-1)
print(f"after sorting: {my_list}")
```


```{admonition} A7
:class: dropdown 

(1) <= <br>
(2) <= <br>
(3) > <br>
(4) arr[pivot], arr[right] = arr[right], arr[pivot] <br>
(5) start <br>
(6) end <br>
```

### Q8. Tree 

Tree is a graph without a (1). There are several properties of tree. State True if the following sentence(s) is correct, State False the sentecne(s) is wrong. 

(2) One of the properties of a tree is there is always a unique path from node U to node V. 
(3) There is a root in a tree. 
(4) Subtree is a partial tree that deleting one edge. 
(5) Child node cannot have children.
(6) There is no need to have a root in a tree. However, if you pick a root and grab it upward, you can visualize the tree starting from the root. 
(7) Assuming there is a root in a tree, then all nodes connected to one node are children of the node except one.

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

Using a property of a tree that all nodes connected to one node is children except one node (parent), we can build a tree in a list form. The following source code is has a function `edge_to_parent_children_list` which takes `edges` list and `N` (the number of nodes) and returns `parents` and `children` list. Fill in the blank so that the code would work after it is filled in. Note that the node number is 1-indexed (starting from 1)

```python
'''
edges to children, parents list 
'''

from typing import List, Tuple 
from collections import deque 

def edges_to_parent_children_list(N:int, edges: List[int], root=1) -> Tuple[List, List]:
    parent = [-1] * (N+1) # 1-indexed 
    children = [[] for _ in range(N+1)]
    graph = [[] for _ in range(N+1)]

    for edge in edges:
        graph[edge[0]].append(edge[1])
        graph[edge[1]].append(edge[0])

    q = deque([root])
    parent[root] = root  # 방문 처리 필수 
    while q:
        cur_val = q.popleft()

        for nxt_node in graph[cur_val]:
            if parent[nxt_node] != -1: # already visited 
                continue 
            (1)
            (2)
            (3)
    return parent, children 


n = 7
edges = [(1,2),(1,3),(2,4),(2,5),(3,6),(3,7)]
root = 1

p, ch = edges_to_parent_children_list(n, edges)
print("parent:", p)       # parent[1]=0, parent[2]=1, parent[3]=1, ...
print("children:", ch)    # children[1]=[2,3], children[2]=[4,5], ...
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

BST is a short term of (1) is a tree is designed for searching a data quickly. The root value of the tree has to be greater than the node(s) valud in the left subtree and the smaller than the value(s) in the right subtree. An interesting charateristic of the BST is that when you traverse it with (4), the resulting array is in ascending order. The time complexity for this appraoch with the number of data N is (5). 

````{admonition} A10
:class: dropdown 

(1) Binary Search Tree <br>
(2) greater <br>
(3) smaller <br>
(4) in-order DFS <br> 
(5) O(logN) <br> 
````

### Q11. Graph Algorithm 

A graph consists of vertices and edges. Especially we learned about (1), where no common element exists in both two sets. It is also called as (2). Two core operations for this data structure is (3) and (4), the former one is to search the parent or root of the node and the latter one is to join the two sets. 

````{admonition} A11
:class: dropdown 

(1) Disjoint sets <br>
(2) Union-Find Data Structure<br>
(3) Find <br>
(4) Union<br>
````

### Q12. Kruskal's Algorithm  

Kruskal's algorithm is an algorithm to find out (1) in a graph. It is a type of (2) algorithm since the algorithm picks what appears to be the best options at the moment. We learned about the algorithm in the class. Descibe the algorithm steps.

````{admonition} A12
:class: dropdown 

(1) MST (Minimum Spanning Tree) <br>
(2) Greedy <br>
(3) 
3-1. Sort the edges in ascending order 
3-2. Pop the minimum weight edge
3-3. Add the weight to the total cost if there is no cycle after adding the edge to the current MST 
3-4. Repeat the 3-2 and 3-3 steps until there is no nodes unvisited 
````
### Q13. Topological Sort 

Topological sort is a sort algorithm, only can be used under the (1). (1) is a tree with directed edges. (2) is a way to solve this topological sort problem, using indegree concepts. Indegree is (3). If there is a node with indegree 0, the node can be thought as a (4) of the result sorted array. This algorithm can be used to detect a cycle in a graph. When we run the algorithm and encounter the situation where (5), then we can tell there is a cycle in the graph. Time complexity for this algorithm is (6), where V is the number of vertices and E is the number of edges. 

Give me at least 3 examples of topological sorts in real life. (7) 


````{admonition} A13
:class: dropdown 

(1) Directed Acyclic Graph (DAG) <br>
(2) Kahn's Algorithm <br>
(3) the number of edges pointing to the node <br> 
(4) starting node <br>
(5) there is no nodes with indegree 0 but are still nodes unvisited <br>
(6) O(V+E) <br>
(7) 1. Prerequisites subjects 2. Compilation of files 3. Waiting Queues <br>
````

### Q14. Topological Sort Code 

Fill in the blanks (blank(1), blank(2), ...etc) to make the code work. 

```python 
from typing import List 
from collections import deque 

def topological_sort(N: int, edges:List[int]):
    '''
    N: the number of nodes 
    edges: edges in the graph 
    Note that the graph is 1-indexed. 
    '''

    indegree = [0] * (N+1)
    graph = [[] for _ in range(N+1)]

    for edge in edges: # edge[0]: starting point, edge[1]: end point 
        graph[edge[0]].append(edge[1])
        graph[edge[1]].append(edge[0])
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

## Problem solving problems 

### Q1. Build Tree 

Implement `build_tree(arr: List) -> Optional[Node]` and `root_to_list(root:Node) -> List[int]` in the following source code. 

```{python}
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
### Q2. Greedy 