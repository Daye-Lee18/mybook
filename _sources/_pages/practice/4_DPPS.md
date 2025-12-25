---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.11.5
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---
# Lecture 4-2. DP 실습

들어가기전, DP 문제를 풀 때 아래 정보들에 유념해서 문제를 푼다면 훨씬 도움이 될 것이다. 

````{admonition} Things to think for a DP problem 
:class: important 

```{code-block} python 
# 6개 checklist 
# 1) State: dp[...]
# 2) What to store: min/max/count/bool/value (필요시 prev/choice)
# 3) Base case init
# 4) Fill order: 작은 상태 → 큰 상태 (Top-down/Bottom-up)
# 5) Transition
# 6) Read answer

def solve(...):
    # 예: 2D (i, j)
    dp = [[INF]* (n+1) for _ in range(m+1)]
    # base cases
    dp[0][0] = 0

    for i in range(0, m+1):
        for j in range(0, n+1):
            if i>0:
                dp[i][j] = min(dp[i][j], f(dp[i-1][j], ...))
            if j>0:
                dp[i][j] = min(dp[i][j], g(dp[i][j-1], ...))

    return dp[m][n]
```
````

- DP 고득점 Kit 
  - [N으로 표현](https://school.programmers.co.kr/learn/courses/30/lessons/42895)
  - [정수 삼각형](https://school.programmers.co.kr/learn/courses/30/lessons/43105)
  - [등굣길](https://school.programmers.co.kr/learn/courses/30/lessons/42898)
  - [사칙연산](https://school.programmers.co.kr/learn/courses/30/lessons/1843)
  - [도둑질](https://school.programmers.co.kr/learn/courses/30/lessons/42897)

- Tree DP 
  - [서브트리에 포함된 정점의 개수 세기](https://www.acmicpc.net/problem/15681): [정답](https://wikidocs.net/272872)
  - [우수 마을](https://www.acmicpc.net/problem/1949): [정답](https://wikidocs.net/274552)
  - [사회망 서비스](https://www.acmicpc.net/problem/2533): [정답](https://wikidocs.net/273100)
  - [Diameter of Binary Tree](https://leetcode.com/problems/diameter-of-binary-tree/description/)
  - [Longest Path with Different Adjacent Character](https://leetcode.com/problems/longest-path-with-different-adjacent-characters/description/)
  - [Binary Tree Max Path Sum](https://leetcode.com/problems/binary-tree-maximum-path-sum/description/)
  - [Difference Between Maximum and Minimum Price Sum](https://leetcode.com/problems/difference-between-maximum-and-minimum-price-sum/description/)
  - [House Robber 3](https://leetcode.com/problems/house-robber-iii/description/)
  - [Maximum Sum BST in Binary Tree](https://leetcode.com/problems/maximum-sum-bst-in-binary-tree/description/?envType=problem-list-v2&envId=50v8rtm7)
  - [Binary Tree Cameras](https://leetcode.com/problems/binary-tree-cameras/description/?envType=problem-list-v2&envId=50v8rtm7)
  - [Kth Ancestor of a Tree Node](https://leetcode.com/problems/kth-ancestor-of-a-tree-node/?envType=problem-list-v2&envId=50v8rtm7)
  - [Find the Shortest Superstring](https://leetcode.com/problems/find-the-shortest-superstring/description/?envType=problem-list-v2&envId=50v8rtm7)
  - [Number of Ways to Reorder Array to Get Same BST](https://leetcode.com/problems/number-of-ways-to-reorder-array-to-get-same-bst/description/?envType=problem-list-v2&envId=50v8rtm7)
- 코테 기출 
  - [색깔 트리](https://www.codetree.ai/ko/frequent-problems/samsung-sw/problems/color-tree/description)
  - [코드트리 메신저](https://www.codetree.ai/ko/frequent-problems/samsung-sw/problems/codetree-messenger/description)
  - [String Compression II](https://leetcode.com/problems/string-compression-ii/description/)


## TreeDP

### 서브트리의 정점 개수 세기 
````{admonition} Tree DP에서 Bottom-up 방식으로 구현하는 이유
:class: dropdown 

- **Tree DP는 보통 Bottom-up(post-order DFS)으로 구현한다.**  
  → 자식들의 값을 모두 구한 뒤 부모 값을 계산하는 구조이기 때문.  
- 경우에 따라 부모 정보를 자식에게 넘겨야 하는 문제는 Top-down(pre-order DFS)을 병행하기도 한다.  
- 따라서 Tree DP에서는 보통 `dfs(curr, parent)` 형태로 구현해,  
  parent-child 관계를 유지하며 DP 값을 계산한다.  
- 또한 기본적으로 "parent"와 "children"정보는 그래프를 받을 때 저장을 해놓야하는 정보이다. 

기본 post-order dfs는 다음과 같다. 

```{code-block} python
# u: 현재 노드, p: parent node 
def dfs(u, p):
    dp[u] = 1   # 자신도 자신을 루트로 하는 서브트리에 포함되므로 0이 아닌 1에서 시작한다.
    for v in graph[u]:
        # tree에서 현재 노드와 연결되어 있는 노드 중 p (부모)빼고는 모두 children이다.
        if v == p:
            continue
        dfs(v, u)             # 자식 처리
        dp[u] += dp[v]        # 자식 값 합치기
```

아래의 그림처럼, acyclic graph를 트리로 변환하는 함수를 구현할 수 있다. 

![1](../../assets/img/DPPS/1.png)

```{toggle}
'''
예시) 
9 5 8
1 3
4 3
5 4
5 6
6 7
2 3
9 6
6 8

'''
n, r, e = map(int, input().split())

graph = [[] for _ in range(n+1)]
children_list = [[] for _ in range(n+1)]
parent_list = [0] * (n+1)
visited = [False] * (n+1)
for _ in range(e):
    a, b= map(int, input().split())
    graph[a].append(b)
    graph[b].append(a)

# parent node가 r인경우 tree로 만들기, top-down (pre-order dfs)
def makeTree(cur_node, parent):
    visited[cur_node] = True 

    for node in graph[cur_node]:
        if node != parent and not visited[node]:
            makeTree(node, cur_node)
            children_list[cur_node].append(node)
            parent_list[node] = cur_node

# root에서 시작 
makeTree(r, 0)

print(f"parent list: {parent_list}")
print(f"children list: {children_list}")

```
````


````{admonition} makeTree and count_subtree
:class: dropdown 

```{code-block} python
# f = open('Input.txt', 'r')
n, r, q = map(int, input().split())

graph = [[] for _ in range(n+1)]
children_list = [[] for _ in range(n+1)]
parent_list = [0] * (n+1)
visited = [False] * (n+1)
dp = [0] * (n+1)

for _ in range(n-1):
    # print(_)
    a, b= map(int, input().split())
    graph[a].append(b)
    graph[b].append(a)

# parent node가 r인경우 tree로 만들기, top-down (pre-order dfs)
def makeTree(cur_node, parent):
    visited[cur_node] = True 

    for node in graph[cur_node]:
        if node != parent and not visited[node]:
            makeTree(node, cur_node)
            children_list[cur_node].append(node)
            parent_list[node] = cur_node

# root에서 시작 
makeTree(r, 0)

# print(f"parent list: {parent_list}")
# print(f"children list: {children_list}")
def count_subtree(cur_node, parent):
   dp[cur_node] = 1 

   for node in children_list[cur_node]:
       count_subtree(node, cur_node)
       dp[cur_node] += dp[node]

count_subtree(r, 0)

for _ in range(q):
    root = int(input())
    print(dp[root])
    
```
````

````{admonition} solution
:class: dropdown 

문제는 위에처럼 maketree와 count_subtree를 따로 만들면, 두번의 dfs를 거쳐야해서 타임 아웃이 된다. 이미 데이터를 받을 때, graph정보안에 하나의 parent빼고 모두 children을 담고 있으므로 이를 이용하여, maketree함수를 따로 만드는 대신, count_tree함수만 사용할 수 있다. 

```{code-block} python 
import sys
sys.setrecursionlimit(1_000_000)
input = sys.stdin.readline

n, r, q = map(int, input().split())

g = [[] for _ in range(n + 1)]
for _ in range(n - 1):
    a, b = map(int, input().split())
    g[a].append(b)
    g[b].append(a)

dp = [0] * (n + 1)
parent = [0] * (n + 1)

def dfs(u, p):
    parent[u] = p
    size = 1
    for v in g[u]:
        if v == p:
            continue
        size += dfs(v, u)
    dp[u] = size
    return size

dfs(r, 0)

out = []
for _ in range(q):
    u = int(input())
    out.append(str(dp[u]))
print("\n".join(out))

```
````

### 우수 마을 

````{admonition} Top-down Solution 
:class: dropdown 

이것을 DP로 구현하기 위해서 두 가지 경우를 생각하면 됩니다. 자신이 우수 마을인 경우와, 자신이 일반 마을인 경우 입니다.
첫 번째로 자신이 우수 마을이라면 자신의 자식 마을은 무조건 일반 마을이어야 합니다. 우수 마을끼리는 인접할 수 없기 때문에 자식 마을이 우수 마을일 수는 없습니다. 반대로 자신이 일반 마을이라면 자식 마을이 꼭 우수 마을일 필요는 없습니다. 부모 마을이 우수 마을이면 되기 때문 입니다.

![2](../../assets/img/DPPS/2.png)

위 그림에서 회색 우수 마을, 파란색은 흰색을 마을이라 생각해 보겠습니다. 1번이 우수 마을이라면 자신의 자식 마을인 2번은 반드시 일반 마을이어야 합니다. 반대로 2번이 일반 마을인 경우에는 자식 마을 3번, 6번이 우수 마을일 필요는 없습니다. 이와 같은 규칙으로 우리는 인구수가 최대한 많은 경우만 따져주면 됩니다.

즉, 아래의 규칙을 따르는 프로그램을 구현하면 된다. 
- 현재 노드가 우수마을이면 자식 마을은 반드시 일반 마을 
- 현재 노드가 일반 마을이라면, 자식 마을은 일반 마을/우수 마을 둘 다 가능 

- state: dp[node] = the node id 
- what to store: (현재 노드가 우수 마을인 경우 'subtree'의 전체 인구수 , 현재 노드가 일반 마을인 경우 'subtree'의 전체 인구수)
- init: (num people of the node, 0)
- recurrence relation: 
    dp[parent][0] += dp[child][1]
    dp[parent][1] += max(dp[child][0], dp[child][1])

```{literalinclude} ../solutions/DPPS/2.py
:language: python
```
````

### 사회망 서비스 

````{admonition} Idea 
:class: dropdown 

![](../../assets/img/DPPS/3.png)

Top-down으로 풀 경우, 트리에 3개의 노드가 있는 경우 위의 그림처럼 경우의수를 구할 수 있다. 
즉, 현재 노드가 early adopter여야만 하는 경우는 딱 한 가지 경우이다. 
반면, 현재 노드가 early adopter인 경우에는 children노드가 어떤 상황인지 전혀 상관없고, 그저 최소한의 경우를 가지고 오면 된다. 

그러나, 재귀함수를 이용해 답안을 적으면, 겨우 통과되는데, 파이썬으로는 재귀를 이용하여 문제를 풀면, 잘 메모리나 재귀횟수때문에 통과가 안되는 일이 많다. 따라서, 위의 DFS 방식을 BFS 방식으로 바꿔야한다. 

<DFS를 BFS로 바꾸는 스텝> 
- 먼저 BFS로 모든 트리를 탐색하며 부모와 자식 노드들을 stack에 넣어준다. 
- `stack.pop()`을 하면서 자연스럽게 말단 노드의 값부터 가져온다. 
- DFS()의 로직을 그대로 따라한다. 
````

````{admonition} DFS Solution 
:class: dropdown 

```{literalinclude} ../solutions/DPPS/3.py
:language: python 
````
````{admonition} DFS Solution 
:class: dropdown 

```{literalinclude} ../solutions/DPPS/3.py
:language: python 
````

````{admonition} BFS Solution 
:class: dropdown 

```{literalinclude} ../solutions/DPPS/3_2.py
:language: python 
````

### Diameter of Binary Tree

````{admonition} Solution
:class: dropdown 

```{literalinclude} ../solutions/DPPS/4.py
:language: python 
```
````

### Longest Path with Different Adjacent Character

````{admonition} Solution 
:class: dropdown 

```{literalinclude} ../solutions/DPPS/5.py
:language: python 
```

````
### Binary Tree Max Path Sum 

````{admonition} Solution 
:class: dropdown 

```{literalinclude} ../solutions/DPPS/6.py
:language: python 
```
````

### Difference Between Maximum and Minimum Price Sum 

````{admonition} Explanation 
:class: dropdown 

이 문제를 트리 + DP(DFS) + rerooting 으로 풀면 O(N*N)으로 Time out 된다. 

관찰
- "한쪽 끝을 제외한다" -> 결국 경로의 '내부 노드' 합을 계산하는 문제 
- 방향성 없음 -> 트리 DP가능 
````

````{admonition} Solution 
:class: dropdown 

```{literalinclude} ../solutions/DPPS/7.py
:language: python 
```
````

### House Robber 3 

````{admonition} Solution
:class: dropdown 

```{literalinclude} ../solutions/DPPS/8.py
:language: python
```
````

### Maximum Sum BST in Binary Tree 

````{admonition} Solution 
:class: dropdown 

```{literalinclude} ../solutions/DPPS/9.py
:language: python 
````

### Binary Tree Cameras 

````{admonition} Solution 
:class: dropdown 

```{literalinclude} ../solutions/DPPS/10.py
:language: python
```
````

````{admonition} Solution2
:class: dropdown 

```{literalinclude} ../solutions/DPPS/10-2.py
:language: python
```
````

### Kth Ancestor of a node 

````{admonition} Solution 
:class: dropdown 

```{literalinclude} ../solutions/DPPS/11.py
:language: python
````

## 코테 기출 

### 색깔 트리 

````{admonition} Explanation
:class: dropdown 

```{literalinclude} ../solutions/DPPS/12_ex.py
:language: python 
````

````{admonition} Solution 
:class: dropdown 

```{literalinclude} ../solutions/DPPS/12.py
:language: python 
````



## 3번: String Compression II 

````{admonition} Problems
:class: dropdown 

```{literalinclude} ../solutions/DPPS/7.md
:language: md 
```
````

