# Midterm 

Submit your answer folder to `gbyou1694@gmail.com`.

## Short answer problems 

- For each question below, fill in the blanks labeled (1), (2), (3), …, or briefly describe what is being asked.
- Write all short-answer responses in a file named `answer.md` and include it in your submission.
- Each correctly answered blank (e.g., (1), (2)) is worth 2 points.

Q1. Complexity and Memory <br>
Each integer occupies (1) bytes of memory. Therefore, if the total memory usage limit is 256 MB, the maximum size of N should be approximately between (2) and (3). Also, in Python, about (4) operations can be executed per second. If you are solving a problem where N = 10,000 and the time limit is 1 second, the maximum allowable algorithm time complexity is (5).

참고) When writing in a .md file, to express exponents such as $2^3$, use the caret `^` symbol inside math mode, e.g., write `$2^3$`. Alternatively, you may write 2**3 for simplicity.

A1. <br>
(1) 4B  <br>
(2) $10^7$  <br>
(3) $10^8$  <br>
(4) 2 * $10^7$ <br>
(5) O(NlogN) <br>

Q2. Graph Representation <br>
A graph can be represented in two main ways: one method stores the edge weights between nodes i and j in a (1), and the other connects each node i to its neighbors [j, k, …] like a linked list, called the (2). The advantage of the adjacency list is that it uses less (3), but its disadvantage is that (4).

A2.  <br>
(1) adjacent matrix  <br>
(2) adjacent list  <br>
(3) memory usage  <br>
(4) it takes longer to find connected nodes  <br>

Q3. Backtracking<br>
Backtracking is a method of systematically finding possible solutions by following given conditions. Unlike (1), it prunes search paths early by skipping any candidate that does not meet specific conditions. In backtracking, (2) represents the number of elements visited so far, and (3) represents the number of different options available at each step. For example, when arranging all possible orders of 2 people selected from 5, the maximum (2) is (4), and the (3) is (5).

A3.  <br>
(1) brute force <br>
(2) depth <br>
(3) branch <br>
(4) 2 <br> 
(5) 5 <br>

Q4. Dynamic programming<br>
Dynamic Programming (DP) is a method of solving large problems by breaking them into smaller subproblems, where the solution to each small problem must remain valid when combined into the larger problem. By using (1), redundant calculations are avoided.

When designing a DP solution, you must define the three key components below:

1. state: **(2)**
2. what to store: **(3)**
3. transition: **(4)**
   
A4. <br>
(1) memoization (for top-down) / tabulation (for bottom-up) <br>
(2) 
(3) 
(4) 

Q5. Two algorithms for solving shortest path problems learned this semester are (1) and (2). The first algorithm finds the shortest path from one node to all others, following a greedy approach—at each step, selecting the node with the smallest current path cost.
For an advanced Python implementation, use (3) with a min-heap structure. The standard Python library that provides this functionality is (4).

Q6. (1) is a type of dynamic programming algorithm. It computes the minimum cost between all pairs of nodes. Implementation typically uses three nested loops. In the DP table, the cost from a node to itself (i → i) is initialized to (2), the cost between unconnected nodes is (3), and directly connected edges are initialized with their given weights. In the outermost loop (variable k), each node is considered as an intermediate node, and if a shorter path is found through k, the value is updated accordingly.


Q7. BFS vs. DFS <br>
BFS stands for (1) and is implemented using a (2) data structure. In contrast, DFS stands for (3) and is typically implemented using (4) through function recursion. Since recursive calls in DFS follow a stack structure, functions are executed and then return in LIFO order. BFS is advantageous for finding the shortest path, while DFS explores deeper paths, making it useful for exploring deep structures in trees and graphs.

Q8. Complexity Ordering <br>
Arrange the following time complexities in order from fastest to slowest:
1. O($N^2$)
2. O(NlogN)
3. O($2^n$)
4. O(N)
5. O(logN)

Q9. Heap<br>
In a min-heap, the smallest value is located at the root. The Python standard library used for heap implementation is (1). This library provides two core methods: (2) for inserting an element and (3) for retrieving the smallest-priority element. When inserting, if the parent node’s value is greater than the new item’s value, the new item “bubbles up” (sift-up) to maintain the heap property. When deleting, the last element of the heap is moved to the root position, and it is compared with its two child nodes; if the smaller child is less than the current value, they are swapped in a sift-down operation to restore the heap property.

## Problem solving problems 

- The following problems cover the material taught up to the midterm, and are all easy-level coding exercises.
- Each problem includes 5 test cases. During grading, additional hidden test cases will be used, so you are encouraged to test your solutions with custom edge cases before submission.
- Save each solution as p1.py, p2.py, p3.py, p4.py, or p5.py within your submission folder.
- Use the provided starter code under the first toggle of each problem as your base implementation.
- Problems p1–p4 are each worth 10 points, and p5 is worth 20 points.

`````{admonition} p1 simulation 
:class: dropdown 

Robot Return to Origin 

There is a robot starting at the position `(0, 0)`, the origin, on a 2D plane. Given a sequence of its moves, judge if this robot ends up at (0, 0) after it completes its moves.

You are given a string moves that represents the move sequence of the robot where moves[i] represents its ith move. Valid moves are 'R' (right), 'L' (left), 'U' (up), and 'D' (down).

Return true if the robot returns to the origin after it finishes all of its moves, or false otherwise.

Note: The way that the robot is "facing" is irrelevant. 'R' will always make the robot move to the right once, 'L' will always make it move left, etc. Also, assume that the magnitude of the robot's movement is the same for each move.


Example 1:

Input: moves = "UD"
Output: true
Explanation: The robot moves up once, and then down once. All moves have the same magnitude, so it ended up at the origin where it started. Therefore, we return true.
Example 2:

Input: moves = "LL"
Output: false
Explanation: The robot moves left twice. It ends up two "moves" to the left of the origin. We return false because it is not at the origin at the end of its moves.
 

Constraints:

1 <= moves.length <= 2 * $10^4$
moves only contains the characters 'U', 'D', 'L' and 'R'.


```{code-block} python
---
caption: source code
---
class Solution:
    def judgeCircle(self, moves: str) -> bool:    
```


```{toggle} 
test case 1 <br>
input: 'UD'<br>
output: True 
```

```{toggle} 
test case 2 <br>
input: 'LL'<br>
output: False 
```
```{toggle} 
test case 3 <br>
input: 'RULD'<br>
output: True 
```
```{toggle}
test case 4 <br>
input: 'RRRRLLLL'<br>
output: True 
```
```{toggle} 
test case 5 <br>
input: "UUDDLRLR"<br>
output: True 
```
`````

`````{admonition} p2 BFS 
:class: dropdown 

Given the root of a binary tree, invert the tree, and return its root.


```{code-block} python
---
caption: source code for p2 
---
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def invertTree(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
```

```{toggle} 
test case 1 <br>
![test1](../../assets/img/midterm/1.png)
input: root = [4,2,7,1,3,6,9] <br>
output: [4,7,2,9,6,3,1]
```
```{toggle} 
test case 2 <br>
![test2](../../assets/img/midterm/2.png)
input: root = [2,1,3] <br>
output: [2,3,1]
```
```{toggle} 
test case 3 <br>
input: root = [] <br>
output: []
```
```{toggle}
test case 4 <br>
input: [1,2,null,3,null] <br>
output: [1,null,2,null,3]
```
```{toggle}
test case 5 <br>
input: [1,2,3,4,5,null,6] <br>
output: [1,3,2,6,null,5,4]
```
`````

`````{admonition} p3 Backtracking
:class: dropdown 

Given the root of a binary tree, return all root-to-leaf paths in any order.

A leaf is a node with no children.


```{code-block} python 
---
caption: source code for p3 
---
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def binaryTreePaths(self, root: Optional[TreeNode]) -> List[str]:
```


```{toggle} 
test case 1 <br>
![3](../../assets/img/midterm/3.png)
input: root = [1,2,3,null,5] <br>
output:["1->2->5","1->3"]
```
```{toggle}
test case 2 <br>
Input: root = [1] <br>
Output: ["1"]
```
```{toggle}
test case 3 <br>
input: [1,2,3,4,5,6] <br>
output: ["1->2->4","1->2->5","1->3->6"]
```
```{toggle}
test case 4 <br>
input: []<br>
output: []
```
```{toggle}
test case 5 <br>
input: [1,2,null,3,null,4,null]<br>
output: ["1->2->3->4"]
```
`````

`````{admonition} p4 DP
:class: dropdown 

You are given an integer array cost where cost[i] is the cost of ith step on a staircase. Once you pay the cost, you can either climb one or two steps.

You can either start from the step with index 0, or the step with index 1.

Return the minimum cost to reach the top of the floor.


```{code-block} python 
---
caption: source code for p4 
---
class Solution:
    def minCostClimbingStairs(self, cost: List[int]) -> int:
        
```

```{toggle}
test case 1 <br>
Input: cost = [10,15,20] <br>
Output: 15 <br>

Explanation: You will start at index 1.
- Pay 15 and climb two steps to reach the top.
The total cost is 15.
```
```{toggle}
test case 2 <br>
Input: cost = [1,100,1,1,1,100,1,1,100,1] <br>
Output: 6 <br>

Explanation: You will start at index 0.
- Pay 1 and climb two steps to reach index 2.
- Pay 1 and climb two steps to reach index 4.
- Pay 1 and climb two steps to reach index 6.
- Pay 1 and climb one step to reach index 7.
- Pay 1 and climb two steps to reach index 9.
- Pay 1 and climb one step to reach the top.
The total cost is 6.
```
```{toggle}
test case 3 <br>
input: [0,0,0,0] <br>
output: 0
```
```{toggle}
test case 4 <br>
input: [5,4,3]<br>
output: 4
```
```{toggle}
test case 5 <br>
input: [1,2] <br>
output: 1
```
`````

`````{admonition} p5 Shortest Path 
:class: dropdown

There are n cities numbered from 0 to n-1. Given the array edges where edges[i] = [fromi, toi, weighti] represents a bidirectional and weighted edge between cities fromi and toi, and given the integer distanceThreshold.

Return the city with the smallest number of cities that are reachable through some path and whose distance is at most distanceThreshold, If there are multiple such cities, return the city with the greatest number.

Notice that the distance of a path connecting cities i and j is equal to the sum of the edges' weights along that path.


```{code-block} python
---
caption: source code for p5
---
class Solution:
    def findTheCity(self, n: int, edges: List[List[int]], distanceThreshold: int) -> int:
        
```

```{toggle}
test case 1 <br>
Input: n = 4, edges = [[0,1,3],[1,2,1],[1,3,4],[2,3,1]], distanceThreshold = 4 <br>
Output: 3 <br>

![4](../../assets/img/midterm/4.png)
Explanation: The figure above describes the graph. 
The neighboring cities at a distanceThreshold = 4 for each city are:
City 0 -> [City 1, City 2] 
City 1 -> [City 0, City 2, City 3] 
City 2 -> [City 0, City 1, City 3] 
City 3 -> [City 1, City 2] 
Cities 0 and 3 have 2 neighboring cities at a distanceThreshold = 4, but we have to return city 3 since it has the greatest number.
```
```{toggle}
test case 2 <br>
Input: n = 5, edges = [[0,1,2],[0,4,8],[1,2,3],[1,4,2],[2,3,1],[3,4,1]], distanceThreshold = 2 <br>
Output: 0 <br>

![5](../../assets/img/midterm/5.png)
Explanation: The figure above describes the graph. 
The neighboring cities at a distanceThreshold = 2 for each city are:
City 0 -> [City 1] 
City 1 -> [City 0, City 4] 
City 2 -> [City 3, City 4] 
City 3 -> [City 2, City 4]
City 4 -> [City 1, City 2, City 3] 
The city 0 has 1 neighboring city at a distanceThreshold = 2.

```
```{toggle}
test case 3 <br>
input: n = 2, edges = [[0,1,5]], distanceThreshold = 4 <br>
output: 1
```
```{toggle}
test case 4 <br>
input: n = 3, edges = [[0,1,1],[1,2,1],[0,2,1]], distanceThreshold = 2 <br>
output: 2
```
```{toggle}
test case 5 <br>
input: n = 6, edges = [[0,1,1],[1,2,1],[2,3,1],[3,4,1],[4,5,1]], distanceThreshold = 2 <br>
output: 5
```
`````