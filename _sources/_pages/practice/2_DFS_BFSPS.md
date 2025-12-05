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

# Lecture 2-2. DFS/BFS 실습

예시 문제 링크 
- BFS 고득점 Kit 
  - [타겟 넘버](https://school.programmers.co.kr/learn/courses/30/lessons/43165)
  - [네트워크](https://school.programmers.co.kr/learn/courses/30/lessons/43162)
  - [게임 맵 최단거리](https://school.programmers.co.kr/learn/courses/30/lessons/1844)
  - [단어 변환](https://school.programmers.co.kr/learn/courses/30/lessons/43163)
  - [아이템 줍기](https://school.programmers.co.kr/learn/courses/30/lessons/87694)
  - [여행 경로](https://school.programmers.co.kr/learn/courses/30/lessons/43164)
  - [퍼즐 조각 채우기](https://school.programmers.co.kr/learn/courses/30/lessons/84021)

- BFS 
  - [AI 로봇 청소기](https://www.codetree.ai/ko/frequent-problems/samsung-sw/problems/ai-robot/description)
  - [미생물 연구](https://www.codetree.ai/ko/frequent-problems/samsung-sw/problems/microbial-research/description)
  - [민트초코 우유](https://www.codetree.ai/ko/frequent-problems/samsung-sw/problems/mint-choco-milk/description)
  - [메두사와 전사들](https://www.codetree.ai/ko/frequent-problems/samsung-sw/problems/medusa-and-warriors/description)
  - [미지의 공간 탈출](https://www.codetree.ai/ko/frequent-problems/samsung-sw/problems/escape-unknown-space/description)
  - [마법의 숲 탐색](https://www.codetree.ai/ko/frequent-problems/samsung-sw/problems/magical-forest-exploration/description)
  - [블록 이동하기](https://school.programmers.co.kr/learn/courses/30/lessons/60063)
  - [고대 문명 유적 탐사](https://www.codetree.ai/ko/frequent-problems/samsung-sw/problems/ancient-ruin-exploration/description)
  - [포탑 부수기](https://www.codetree.ai/ko/frequent-problems/samsung-sw/problems/destroy-the-turret/description)
  - [코드트리 빵](https://www.codetree.ai/ko/frequent-problems/samsung-sw/problems/codetree-mon-bread/description)
  - [색깔 폭탄](https://www.codetree.ai/ko/frequent-problems/samsung-sw/problems/colored-bomb/description)
  - [회전하는 빙하](https://www.codetree.ai/ko/frequent-problems/samsung-sw/problems/rotating-glacier/description)
  - [자율주행전기차](https://www.codetree.ai/ko/frequent-problems/samsung-sw/problems/autonomous-electric-car/description)
  - [전투로봇](https://www.codetree.ai/ko/frequent-problems/samsung-sw/problems/fighting-robot/description)
  - [토스트 계란틀](https://www.codetree.ai/ko/frequent-problems/samsung-sw/problems/toast-eggmold/description)

## BFS 고득점 Kit 

### 타겟 넘버 

````{admonition} Explanation
:class: dropdown 

n개의 음이 아닌 정수
순서를 바꾸지 않고 적절히 더하거나 빼서 타겟 넘버를 만들려고 한다. 

parameters:
- numbers: 사용할 수 있는 숫자가 담긴 배열 2 <= numbers.length <= 20 
- target: 타겟넘버 1<= target <= 1000

return:
- 타겟 넘버를 만드는 방법의 수를 반환 

DFS를 사용하면 branch = 2, depth =N개라, 최악의 경우 2^20개 
````
````{admonition} Solution
:class: dropdown 

```{code-block} python
def solution(numbers, target):
    
    def dfs(depth: int, total: int):
        nonlocal answer
        # termination 
        if depth == n:
            if total == target:
                answer += 1 
            return 

        # branch + / -
        dfs(depth+1, total+numbers[depth])
        dfs(depth+1, total - numbers[depth])

    answer = 0
    n = len(numbers)
    dfs(0, 0)

    return answer 

# numbers = [1, 1, 1, 1, 1]; target=3 # 5
numbers = [4, 1, 2, 1]; target=4 # 2 
print(solution(numbers, target))
```
````

### 네트워크 

````{admonition} Solution 
:class: dropdown 

union-find 구조를 이용해 풀이 

```{code-block} python 

def find(a):
    global parent 
    if parent[a] == a:
        return a
    parent[a] = find(parent[a]) # path compression 
    return parent[a]

def union(a, b):
    global rank
    rootA = find(a); rootB = find(b)

    if rootA == rootB:
        return False 
    if rank[rootA] == rank[rootB]:
        parent[rootB] = rootA 
        rank[rootA] += 1 
    elif rank[rootA] > rank[rootB]:
        parent[rootB] = rootA 
    else:
        parent[rootA] = rootB 
    return True 
   


def solution(n, computers):
    global parent, rank, GroupCnt 
    GroupCnt = n 
    parent = [idx for idx in range(n)]
    rank = [0] * n # height 

    for node in range(n):
        for end_node in range(node+1, n):
            if computers[node][end_node] == 0:
                continue 
            else: # 연결되어 있고 
                if union(node, end_node):
                    GroupCnt -= 1 

    return GroupCnt 

```
````

### 게임 맵 최단 거리 

````{admonition} Solution 
:class: dropdown 

```{code-block} python 
from collections import deque 


def solution(maps):
    N = len(maps); M = len(maps[0])

    def in_range(y, x):
        nonlocal N, M 
        return 0<=y<N and 0<=x<M 
    DY = [-1, 1, 0, 0]; DX = [0, 0, -1, 1]

    visited = [[False]*M for _ in range(N)]
    start_y = 0; start_x = 0
    target_y = N-1; target_x = M-1 
    q = deque([(start_y, start_x, 1)]) # y, x, dis 
    # BFS 
    while q:
        cur_y, cur_x, cur_dis = q.popleft()

        # Early Stopping 
        if cur_y == target_y and cur_x == target_x:
            return cur_dis 
        
        for t in range(4):
            nxt_y = cur_y + DY[t]
            nxt_x = cur_x + DX[t]
            if in_range(nxt_y, nxt_x) and maps[nxt_y][nxt_x] == 1 and not visited[nxt_y][nxt_x]:
                visited[nxt_y][nxt_x] = True 
                q.append((nxt_y, nxt_x, cur_dis + 1 ))
        
    return -1  # 위에서 도달하지 못한 경우 -1를 return 


# maps= [[1,0,1,1,1],[1,0,1,0,1],[1,0,1,1,1],[1,1,1,0,1],[0,0,0,0,1]] # 11 
maps= [[1,0,1,1,1],[1,0,1,0,1],[1,0,1,1,1],[1,1,1,0,0],[0,0,0,0,1]]	# -1 
print(solution(maps))
```
````

### 단어 변환 

“변환 단계 최소 / 간선 개수 최소 / 최소 횟수” → 무조건 BFS 떠올리는 습관

````{admonition} Idea 
:class: dropdown 

parameters: 
- 두 개의 단어 begin, target
    - 두 단어는 같지 않다. 
- 단어의 집합 `words`, 3 <= words.length <= 50, 각 단어 3 <= str.length <= 10 

아래의 규칙을 이용하여 begin -> target 변환하는 가장 짧은 변환 과정을 찾으려고 한다. 
1. 한 번에 한 개의 알파벳만 바꿀 수 있음 
2. `words`에 있는 단어로만 변환할 수 있음 (target도 `words`안에 있어야 함.)

예를 들어, beging이 "hit"이고 target이 "cog", words = ["hot", "dot", "dog", "lot", "log", "cog"]라면 
"hit" -> "hot" -> "dot" -> "dog" -> "cog"와 같이 4단계를 거쳐 변환할 수 있다. 

return: 
- "최소" 몇 단계의 과정을 거쳐 begin을 target으로 변환할 수 있는 지 return 하도록 solution함수를 작성해주세요. 
- 변환할 수 없을 때 0을 반환 

Idea: 
- 문제에서 "최솟값"을 계산하라고 했으니까, BFS로 접근해보면, 각 단어들을 node로 설정하고
- 하나의 char만 다른 값들만 edge 연결시킨다. 
- 그리고 begin노드에서 target 노드까지 연결된 최소 거리를 측정하면 된다. (shortest path)
````

````{admonition} Solution 
:class: dropdown 

```{code-block} python 
from heapq import heappush, heappop 

MAX = int(1e10)

def is_one_different(word1: str, word2: str) -> bool:
    cnt = 0

    if len(word1) != len(word2):
        return False 
    
    for idx in range(len(word1)):
        if word1[idx] != word2[idx]:
            cnt += 1 
        if cnt >= 2:
            return False 
    return True 

def dijkstra(start_node, target_node):
    global shortest_path 

    shortest_path[start_node] = 0 
    pq = [(0, 0)]
    # (E+V) log(V)
    while pq:
        cur_dis, cur_node = heappop(pq) # VlogV

        if cur_dis > shortest_path[cur_node]:
            continue 

        if cur_node == target_node:
            return 
        
        # Elog(V)
        for nxt_node in graph[cur_node]:
            nxt_dis = cur_dis + 1 
            if nxt_dis < shortest_path[nxt_node]:
                shortest_path[nxt_node] = nxt_dis 
                heappush(pq, (nxt_dis, nxt_node))


def solution(begin, target, words):
    global graph, shortest_path 
    if target not in words:
        return 0
    
    n = len(words) + 1  # words에 있는 단어들 + begin 단어 
    graph = [[] for _ in range(n)]
    
    # graph INIT ~ O(N^2)
    for idx, word in enumerate(words): 
        if is_one_different(begin, word):
            graph[0].append(idx+1) # words안에 있는 단어들은 index +1 (begin이 1증가)
            graph[idx+1].append(0)
        if word == target:
            target_id = idx + 1 
        
        for j in range(idx+1, len(words)):
            if is_one_different(word, words[j]):
                graph[idx+1].append(j+1)
                graph[j+1].append(idx+1)
    # ~ O(ElogN)
    shortest_path = [MAX] * n
    dijkstra(0, target_id)
    return shortest_path[target_id] if shortest_path[target_id] != MAX else 0


# begin = "hit"; target="cog"; words = ["hot", "dot", "dog", "lot", "log", "cog"] # 4 
begin = "hit"; target="cog"; words = ["hot", "dot", "dog", "lot", "log"] # 0
print(solution(begin, target, words))
```
````

### 아이템 줍기 

````{admonition} Idea 
:class: dropdown 

이 문제는 그래프를 직접 adjacency list로 그리려고 하면 괴로워진다. 애초에 "정점-간선 그래프"가 아닌 격자 (grid) + BFS로 생각하는게 훨씬 편하다. 

![](../../assets/img/DFS_BFSPS/12.png)

위의 상황에서 아래처럼 좌표를 2배로 만들면, 모서리와 안쪽 칸이 분리되어 테두리만 정확히 따라갈 수 있게 된다. 

![](../../assets/img/DFS_BFSPS/13.png)


1. 좌표를 2배로 키운다. 
    - 이유: 직사각형들이 꼭짓점만 닿을 때, 대각선으로 잘못 돌아가는 길을 막기 위해 
    - 좌표를 2배로 만들면 "모서리"들이 모두 칸 사이에 생겨서, 테두리만 정확히 따라갈 수 있음 
2. 직사각형들로 2D 맵을 만든다.
    - 처음에는 직사각형 전체 영역을 1(지나갈 수 있음)으로 채운다. 
    - 그 다음, 직사각형 내부 (테두리 제외)는 0 (못지나감)으로 채운다.
    - 이렇게 하면 딱 '테두리'만 1로 남음 -> 우리가 갈 수 있는 길은 이 1인 칸들 
3. 캐릭터 위치에서 아이템 위치까지 BFS 
    - 시작점, 도착점 좌표도 2배로 
    - 상/하/좌/우 네 방향으로만 이동 
    - 1인 칸만 이동 가능 
    - BFS에서 (처음 아이템에 도달했을 때 거리 /2) 가 정답 

왜 이렇게 하면 효율적인가?
- 좌표가 최대 50이라서, 2배를 해도 최대 100 × 100 정도 격자.
- BFS 한 번 돌려도 O(100 * 100) 정도 → 충분히 빠름.
- 직사각형 개수도 많지 않아서, 전체 채우는 것도 O(rectangle 수 * 100 * 100) 이하.
- 그래프의 인접 리스트를 만들 필요 없이, 그냥 2D 배열 + BFS로 끝낼 수 있어서 코드도 훨씬 깔끔해.
````
````{admonition} Solution
:class: dropdown 

한가지 짚을 점은, 아래 구현에서 range체크를 할 때, `0<=nx<=MAX`로 `2<=nx<=MAX`를 사용하지 않았다. x와 y의 범위는 원래 1<= x, y<=50인데, 이처럼 하는 이유는, graph의 유효 인덱스 범위는 행/열 인덱스: 0~101이기 때문이다.

왜냐하면, grahp[0][..], graph[1][..]은 이미 0인 상태이며 뒤의 range는 넉넉히 해두고 실제 갈 수 있는 상태는 graph[0][..], graph[1][..]의 0/1 상태로 판단하기 때문이다. 

```{code-block} python 
from collections import deque 

def solution(rectangle, characterX, characterY, itemX, itemY):
    # N = 51
    # 1. 좌표 2배 스케일링  
    MAX = 102 
    graph = [[0] * MAX for _ in range(MAX)]
    # (x1, y2) ~ (x2, y2)
    # (x1, y1) ~ (x2, y1)
    #2-1. 직사각형 전체를 1로 채우기 
    for x1, y1, x2, y2 in rectangle:
        x1 *= 2 ; y1 *= 2 ; x2 *= 2  ; y2 *= 2
        # 직사각형 안에는 다 1로 채우기 
        for y in range(y1, y2+1):
            for x in range(x1, x2+1):
                graph[x][y] = 1 

    #2-2. 직사각형 내부는 0으로 지워서 테두리만 남기기 
    for x1, y1, x2, y2 in rectangle:
        x1 *= 2 ; y1 *= 2 ; x2 *= 2  ; y2 *= 2
        # 직사각형 안에는 다 1로 채우기 
        for y in range(y1+1, y2):
            for x in range(x1+1, x2):
                graph[x][y] = 0 

    # 3. BFS로 최단 거리 탐색 
    sx, sy = characterX*2 , characterY*2 
    ex, ey = itemX*2 , itemY*2 

    dist = [[-1]*MAX for _ in range(MAX)]
    q = deque()
    q.append((sx, sy))
    dist[sx][sy] = 0

    dx = [1, -1, 0, 0]
    dy = [0, 0, 1, -1]

    while q:
        x, y = q.popleft()

        # 아이템 위치 도달하면 (2배 스케일링 했으니) 거리 나누기 2 
        if x == ex and y == ey:
            return dist[x][y] // 2 
        
        for k in range(4):
            nx = x + dx[k]
            ny = y + dy[k]

            # 맵 범위 확인 + 테두리(1)만 이동 + 미방문
            if 0 <= nx <= MAX and 0 <= ny <= MAX:
                if graph[nx][ny] == 1 and dist[nx][ny] == -1:
                    dist[nx][ny] = dist[x][y] + 1 
                    q.append((nx, ny))



            
# rectangle = [[1,1,7,4],[3,2,5,5],[4,3,6,9],[2,6,8,8]]; s_x = 1; s_y=3; item_x = 7; item_y=8 # 17 
# rectangle = [[1,1,8,4],[2,2,4,9],[3,6,9,8],[6,3,7,7]]; s_x = 9; s_y=7; item_x = 6; item_y=1 # 11
# rectangle = [[1,1,5,7]]; s_x = 1; s_y=1; item_x = 4; item_y=7 # 9
# rectangle = [[2,1,7,5],[6,4,10,10]]; s_x = 3; s_y=1; item_x = 7; item_y=10 # 15
rectangle = [[2,2,5,5],[1,3,6,4],[3,1,4,6]]; s_x = 1; s_y=4; item_x = 6; item_y=3 # 10
print(solution(rectangle, s_x, s_y, item_x, item_y))
```
````

### 여행 경로 

````{admonition} Solution 
:class: dropdown 

graph alogirhtm의 오일러 경로를 공부하면 된다. 

```{code-block} python 
from collections import defaultdict

def solution(tickets):
    '''
    Eulerian Path 
    '''
    stack = []
    graph = defaultdict(list)
    for u, v in tickets:
        graph[u].append(v)

    # Sort
    for node in graph:
        graph[node].sort(reverse=True)

    stack = ["ICN"]
    route = []
    while stack:
        last_node = stack[-1] 
        
        if graph[last_node]:
            stack.append(graph[last_node].pop())
        else:
            route.append(stack.pop())

    return route[::-1]


# tickets = [["ICN", "JFK"], ["HND", "IAD"], ["JFK", "HND"]] # ['ICN', 'JFK', 'HND', 'IAD']
# tickets = [["ICN", "SFO"], ["ICN", "ATL"], ["SFO", "ATL"], ["ATL", "ICN"], ["ATL","SFO"]] # ['ICN', 'ATL', 'ICN', 'SFO', 'ATL', 'SFO']
# tickets = [["ICN", "A"], ["ICN", "B"], ["C", "ICN"], ["B", "D"], ["D", "E"], ["E", "A"], ["A", "C"]] # ['ICN', 'A', 'C', 'ICN', 'B', 'D', 'E', 'A']
tickets = [["ICN", "A"], ["A", "B"], ["B", "ICN"]] # ['ICN', 'A', 'B', 'ICN']
print(solution(tickets))
```
````

### 퍼즐 조각 채우기 

````{admonition} Idea 
:class: dropdown 

- CCW 로 90도 회전하는 경우, 좌표는 다음과 같이 변한다.  (y, x) -> (N-1-x, y)
- 문제를 읽어보면 퍼즐 조각과 비어있는 부분은 같은 사이즈일 경우, 그 size에 해당하는 퍼즐 조각과 비어있는 부분의 "모양"이 같은 지 확인해야한다. 
    - dict[int, list[tuple]]로 채워준다. 
        - key: size 
        - value: key에 해당하는 사이즈를 가진, locs: list[tuple]들의 모임 
    - 모양을 확인하기 위해, cell들의 상대적인 위치를 계산하도록 한다. 
        - 좌상단 (y, x)둘다 Min_heap을 기준점으로 하여 나머지 Locs들을 좌상단 (y,x)에 대해서 subtraction을 해주면, 상대적인 위치로 변한다. 이를 sort()하면, 두 리스트가 동일하다면, 모양이 같은 것으로 판명할 수 있다. 
- 필요한 자료 구조 
    - empty_spaces 
    - puzzles_parts 
    - is_empty_spaces_used 
````

````{admonition} Explanation 
:class: dropdown 

테이블 위에는 "게임 보드"와 "테이블"이 있다. 

- 게임보드와 테이블은 모두 각 칸이 1x1 크기인 정사각 격자 모양(NXN)이다. 
    - 0은 빈칸, 1은 이미 채워진 칸 
    - 게임 보드에 퍼즐 조각이 놓일 빈칸 및 table위의 퍼즐조각은 최소 1개에서 6개까지 연결된 형태 
- 다음 규칙에 따라 "테이블" 위의 퍼즐 조각을 "게임 보드"의 빈칸에 채운다. 

INIT:
- 처음 테이블과 게임보드는 인접한 칸이 닿지 않도록 퍼즐들이 차있다.

규칙 
1. 조각은 한 번에 하나씩 채워 넣는다. 
    - 한 조각이 해당 게임 보드에 들어갈 수 있는 여부 확인 후 다음 조각을 확인해야한다. for loop
    - 문제는, '최대한 많은' 조각을 채워넣어야한다. 
    - 똑같은 빈칸이라도, 1개를 넣느냐, 여러개를 넣을 수 있느냐에 따라 다른데, 이때 좋은 점은 인접한 칸이 비어있지 않아야 하므로, 
        한 번 넣을 때 빈 칸이 온전히 꽉차도록 배치해야한다는 것이다. (규칙 4번에 의해)
2. 조각을 회전시킬 수 있다. 
    - 4개 가능: 그대로 + Rotate(90도, 180도, 270도)
3. 조각을 뒤집을 수는 없다. 
4. "게임 보드"에 "새로" 채워 넣은 퍼즐 조각과 인접한 칸이 비어있으면 안 된다. 

Algorithm 
1. game_board을 bfs: 비어있는 곳 확인 ~ O(N^2)
    - dict[int, [list[(y, x)]]: dict[size]에는 넓이가 size만한 비어있는 위치 정보들이 저장되어 있음. 
2. table 을 bfs: 퍼즐 조각 모양 확인  ~ O(N^2)
    - dict[int, [list[(y, x)]]: dict[size]에는 넓이가 size만한 퍼즐 조각들의 정보들이 저장되어 있음. 

3. 사이즈가 작은 순서대로, puzzles_parts[size]를 탐색. 
- for size 
    for 현재 사이즈 퍼즐 조각 idx_p 
        for 현재 사이즈의 비어있는 공간 idx_e
            if is_empty_spaces_used == True:
                continue 
            if (현재 사이즈 퍼즐 조각을 비어있는 공간에 넣을 수 있다): -> 현재 모양, 90, 180, 270에 대해서 돌렸을 때의 모양까지 확인 
                total += 1 
                break # 넣을 수 있으면 현재 퍼즐 조각에 대해서 여기서 끝내야함 
        


return:
- 규칙에 맞게 최대한 많은 퍼즐 조각을 채워 넣을 경우, 총 몇 칸을 채울 수 있는지 return 하도록 solution 함수를 완성해라.
````
````{admonition} Solution
:class: dropdown 

```{code-block} python
from collections import defaultdict , deque 
from heapq import heappush, heappop 

def in_range(y, x):
    global N
    return 0 <= y < N and 0 <= x < N 

def BFS(s_y, s_x, visited, flag, graph):
    global puzzles_parts, empty_spaces, is_empty_spaces_used
    '''
    floodfill을 하면서, 조건에 맞는 곳을 Visited에 삽입시킨다. 
    '''
    DY= [-1, 1, 0, 0]; DX = [0, 0, 1, -1]
    visited.add((s_y, s_x))

    q = deque([(s_y, s_x)])
    locs = []
    size = 0
    while q:
        cur_y, cur_x = q.popleft()
        # start 위치인 (s_y, s_x)에 대해 "상대적인 위치" 저장 
        locs.append((cur_y - s_y, cur_x - s_x)) 
        size += 1 

        for t in range(4):
            ny = cur_y + DY[t] ; nx = cur_x + DX[t]
            if (ny, nx) in visited:
                continue 

            if in_range(ny, nx) and graph[ny][nx] == flag:
                visited.add((ny, nx))
                q.append((ny, nx))
    
    locs.sort(key=lambda x: (x[0], x[1])) # 정렬 y-min, x-min 정렬 
    if flag == 1: # 채워져있는 칸을 찾는 puzzle parts를 찾는다면, 
        puzzles_parts[size].append(locs[:]) # slicing 
    else: # 비어있는 칸을 찾는 다면, 
        empty_spaces[size].append(locs[:])
        is_empty_spaces_used[size].append(False)

def make_locs_relative_locs(locs):
    '''
    이미 정렬된 Locs를 받음
    in-place로 상대적 위치로 변환 
    '''
    ref_y, ref_x = locs[0]
    for idx in range(len(locs)):
        locs[idx] = (locs[idx][0] - ref_y, locs[idx][1] - ref_x)


def can_fit(puzzle_parts_locs, empty_locs):
    '''
    두 Parameters들은 각 조각/비어있는 곳에 대한 
    (0, 0)에 대한 상대적 위치로 저장되어 있음. 
    '''
    global N
    assert len(puzzle_parts_locs) == len(empty_locs)

    before_locs = puzzle_parts_locs[:]
    for _ in range(4): # puzzle_parts 4번의 rotation 
        # 모든 puzzle_parts의 현재 locs에 대하여 s
        after_locs = []
        for loc in before_locs: # before_locs의 원소양이 바뀌면 안되므로, temp_locs에 대신 채워넣음. 
            y = loc[0]; x = loc[1]
            after_locs.append((N-1-x, y))  
        

        before_locs = after_locs[:] # Update 다음 90도 CCW를 위해 

        # 회전된 좌표들에 대해 좌표들의 '상대적 위치'가 동일한지 확인 
        # empty_locs는 한 번 만들면 변하지 않고, 이미 상대적 위치들이 정렬되어 있음.  
        after_locs.sort(key=lambda x: (x[0], x[1])) 
        make_locs_relative_locs(after_locs)
        if empty_locs == after_locs:
            return True 
    return False 




def solution(game_board, table):
    global empty_spaces, puzzles_parts, is_empty_spaces_used, N
    empty_spaces = defaultdict(list)
    puzzles_parts = defaultdict(list)
    is_empty_spaces_used = defaultdict(list) # False/True 

    N = len(table)

    # Step 1. BFS로 empty_spaces와 puzzles_parts를 size별로 위치 정보 계산 
    empty_spaces_visited = set()
    puzzles_parts_visited = set()
    for y in range(N):
        for x in range(N):
            if not (y, x) in empty_spaces_visited and \
                game_board[y][x] == 0: # 비어져있는 칸이 비어있는 곳 
                BFS(y, x, empty_spaces_visited, 0, game_board)
            if not (y, x) in puzzles_parts_visited and \
                table[y][x] == 1: # 채워져있는 칸이 puzzle parts 
                BFS(y, x, puzzles_parts_visited, 1, table)

    # Step 2. 사이즈 별로, puzzles_parts[size]를 탐색 
    total = 0
    for size in puzzles_parts.keys():
        for puzzle_locs in puzzles_parts[size]: # 모든 퍼즐에 대하여 
            if size in empty_spaces:
                for idx_e, empty_locs in enumerate(empty_spaces[size]):
                    if is_empty_spaces_used[size][idx_e] == True:
                        continue 
                    if can_fit(puzzle_locs, empty_locs):
                        total += size # "총 몇 칸을 채울 수 있는지"
                        is_empty_spaces_used[size][idx_e]= True 
                        break # 찾았으면 맞는 빈 공간 찾는 것을 중단하고, 다음 puzzle 조각으로 넘어가야함. 

                    
    return total 

```
````

## BFS 

### AI 로봇 청소기 


````{admonition} 실수한 부분 
:class: dropdown 

이런 BFS문제를 풀 때, 각 격자가 가질 수 있는 상태의 수가 매우 중요함. 변화하는 상태에 대해서 update할 때 확인해야 할 것이 무엇인지 알 수 있기 때문 

각 격자는 
(1) 먼지가 있거나: 먼지의 양(p) 1 <= graph[y][x] <= 100 로 존재. 
(2) 아무런 먼지가 없거나: graph[y][x] == 0
(3) 물건이 위치할 수 있음: graph[y][x] == -1
(4) 청소기 위치: vaccume_list[id].y, vaccume_list[id].x, locs_vaccume_set: set(tuple)
-> 청소기의 위치를 나타내는 2차원 배열을 따로 만들어서 관리 B[r][c] = robot.num 

이번 문제에서, 가장 큰 오류는 clean() 함수에서 존재하였다. 
- 청소할 '방향'을 결정할 때, 현재 그래프의 먼지량 총합의 기준이 아니라, "청소할 수 있는"먼지 총합을 기준으로 방향을 설정했어야했다. 
- 즉, 한 격자당 20 먼지량을 청소할 수 있으므로 총 합을 구할 때 min(20, A[r][c])로 구해야하는데, 그냥 A[r][c]을 계산해서 틀렸음. 
  
또한 시간을 가장 많이 잡아먹는 부분이 move()의 BFS 함수에서 존재하였다. 
- 현재 로봇 청소기 위치에 먼지가 있으면 움직이지 않아도 된다. 라는 부분이 잘 명시되어 있지 않아서 헷갈렸다. 
- 가장 가까운 오염거리에 대한 위치 계산을 할 때, 모든 오염 셀에 대한 거리를 계산하는 게 아니라, 로봇 위치에서 가장 가까운 격자의 거리를 계산하면 그 다음은 break를 하는 것이 훨씬 빠른데, 전자의 방법을 사용하여 시간 초과 에러를 받았다. 

마지막으로, constraints에 대한 time complexity계산을 어느 정도 한 뒤에 알고리즘을 짜야하는데, 너무 급하게 구현을 한 것 같다. 
````

````{admonition} Solution 
:class: dropdown 

```{literalinclude} ../solutions/DFS_BFSPS/1.py
:language: python 
```
````



###  미생물 연구 

````{admonition} Explanation 
:class: dropdown 

미생물 연구 

필요한 자료구조 
- `graph`: 현재 미생물의 위치를 저장한다. graph[y][x] = microbe_id , 아무것도 없으면 0으로 저장 (INIT)
- `does_id_exist: [bool]` = does_id_exit[id] = True/False로 저장 해당 미생물 id가 죽었는지 살았는지 체크 
- `id_to_locs: dict[int, list]` = 미생물들은 삽입되거나 제거되므로 add()뒤에 move()를 시작할 때 항상 계속 갱신 
- `id_to_new_graph_origin: dict[int, Tuple[int]]`: move()이후에 id마다의 locs 재계산은 time consuming하므로 move()할때 새로 옮긴 origin을 저장해두어, record()때 이웃 미생물 검사시에 사용한다. 

- 배양 크기 NxN (0, 0) ~ (N, N), 0-indexed , N+1개
    - 0 <= x,y < N (inclusive, exclusive)
    - 좌하단이 (0, 0)이며 (x, y)로 표시 

- 총 Q번의 실험 진행하며 각 실험의 결과를 기록 

1. 미생물 투입 (add) ~ O(N) + O(N)
- (r1, c1, r2, c2)의 직사각형 여역에 한 무리의 미생물 투입 
    - graph[y][x] = microbe_id 저장한다. 
    - does_id_exist = [True] # id는 1부터 시작하므로, 맨 앞은 [True]로 저장 
    - does_id_exist.append(True)
- 원래 자리에 다른 미생물이 존재하면 새로 투입된 미생물의 영역 내의 미생물들을 잡아먹는다. 
    - 원래 뭐가 있어도 현재 microbe_id로 저장한다. 
    - 이때 기존에 있는 미생물 id를 `removed_id = set()`에 저장해둔다. 
- 즉, 영역 내에는 새로 투입된 미생물만 남게된다. 
- 만약 기존에 있던 어떤 미생물 무리 A가 새로 투입된 미생물 무리 B에게 잡아먹혀서 영역이 2이상으로 나눠진 경우에는, 기존 미생물은 전부 사라진다. 
    - map을 돌면서 해당 미생물 id의 그룹 수가 2이상이면 모두 제거 가능 
        - `does_id_exist[id]: list[int]` = True/False에서 False 로 저장한다. 
    - X: (아래에서 다시 셀 것임) 만약 그룹 수가 2이상이 아니라면, 기존에 있던 `id_to_locs: list[set]`의 set에서 id_to_locs[id].remove(locs)로 해당 위치를 지워준다. 
    
--> 위의 문제는 뭐냐면, 기존에 있던 배양용기에서 뭔가가 사라지면, pq안에서 순서가 제대로 된 것이 없어지게 된다. 
--> 이 때 또 graph를 돌면서 해당 id에 대해 계산해야하기때문에, step 2 (move)에서 한번에 계산하는게 더 편리할 것 

2. 배양 용기 이동 (move)
- 모든 미생물을 새로운 배양 용기로 이동시킴. (NxN)
    - new_graph를 생성한다. (임시)
- 기존 배양 용기에 있는 무리 중 가장 차지한 영역이 넓은 무리를 하나 선택 -> 동일한 영역이면, 가장 먼저 투입된 미생물을 선택한다. 
    - Priority queue / sort 
    - 기존 graph를 돌면서, (len(locs), id)정보를 저장하고, 이를 pq에 다 넣는다. 
        - 새롭게 INIT된 빈 id_to_locs: dict(int, list)
        - 이 과정에서 `id_to_locs[id].append()`하여 locs 위치 정보를 파악해놓는다. 
        - 위에서 pq에 바로 넣는 식으로하면, 예전에 pq에 미리 넣어놨던 것에서 미생물의 영역이 작아지는 경우에는 tracking하기가 어려우므로 다시 센다. 
        - list.sort()로 한다. 
- 선택된 미생물 무리를 새 배양 용기에 옮긴다. (new_graph)
    - 이때 무리는 기존 용기에서의 "형태를 유지"해야한다. 
        - 해당 무리의 cell 좌표를 모두 가지고 있다가, origin이 바뀌면 그만큼 "평행이동"해준다. 
    - 배양 용기의 범위를 벗어나지 않아야 한다. 
        - in_range() 함수 
    - 다른 미생물의 영역과 겹치지 않도록 두어야한다. 
        - check(): 해당 id_to_locs[id]에 있는 모든 위치에서 new_graph의 모든 지점이 0이여야함. 
    - 위의 조건 안에서 x좌표가 최대한 작은 위치로 미생물을 옮겨야 하며, 그런 위치가 둘 이상이라면 최대한 y좌표가 작은 위치로 오도록 미생물을 옮긴다. 
        - (최대한 '좌측하단의 좌표'가 배양 용기의 좌측하단에 오도록 위치)
        - 미생물 id안에서 좌측하단의 좌표를 항상 알고 있는 것이 좋음 
        - 같은 x에서는 작은 y : for x (for y)
        - 새로 옮겨진 위치는 (기존 위치 - 좌측하단 위치) 
- 위의 조건을 만족하지 못하여 어떤 곳에도 둘 수 없으면, 새 용기에 옮기지 않고 사라진다. 
    - does_id_exist[id] = False 로 저장 

graph = new_graph[:] # 슬라이싱 

3. 실험 결과 기록 (record)
- 미생물 무리 중 상하좌우로 맞닿은 면이 있는 무리끼리는 '인접한 무리'
    - 인접한 무리 쌍을 저장한다. (1, 2), (2, 3)
    - 새로 옮겨진 위치에서 인접한지 확인한다. 
    total = 0
    pair_set = set()
    for id in id_to_locs:
        if does_id_exist[id]:
            id_to_locs[모든 위치] - lower_bottom_loc[id]의 4방향에 다른 id가 있으면, 인접 
                graph[dx][dy] != 0 and graph[dx][dy] != id:
                    인접 
                    if (id1, id2) in pair_set or (id2, id1) in pair_set:
                        continue 
                    total += len(id_to_locs[id1]) * len(id_to_locs[id2])
                    pair_set.add((id1, id2))

- 모든 '인접한 무리' "쌍" 을 확인한다. 
- (A,B)와 (B,A)는 같은 쌍임 
- 미생물 A의 영역의 넓이 x 미생물 B의 영역의 넓이만큼의 성과를 얻고 -> 이를 다 "더한" 값 (누적합)을 기록해야한다. 


constraints: 
- 2 <= N <= 15 
- 1 <= Q <= 50 
````

````{admonition} Solution 
:class: dropdown 

```{literalinclude} ../solutions/DFS_BFSPS/2.py
:language: python 
```
````

````{admonition} test case 
:class: dropdown
1. 
8 4 <br>
2 2 5 6<br>
2 3 5 8<br>
2 0 5 3<br>
1 1 6 6<br>

2.
3 3<br>
0 0 3 2 <br>
1 0 3 3<br>
2 2 3 3<br>

3. 
8 5<br>
0 0 2 5<br>
1 1 5 6<br>
1 0 3 6<br>
7 7 8 8<br>
2 4 6 7<br>

4. 
6 6 <br>
3 4 5 5<br>
3 3 4 4<br>
2 0 3 1<br>
0 2 1 4<br>
1 4 3 6<br>
5 0 6 2<br>
````


### 민트초코 우유 

````{admonition} Solution 
:class: dropdown 

```{literalinclude} ../solutions/DFS_BFSPS/3.py
:language: python 
```
````

````{admonition} list.sort() vs. heapq  
:class: dropdown 

어디서 틀렸나? 사실 잘 모르겠음. 

중요한 건, 보통의 경우는 list.sort()를 사용하고 아래와 같은 경우에만 min-heap을 쓴다. 왜냐면 min-heap의 pop()이후 다시 heapify하는데 시간이 걸리기 때문. 

- 간선이 스트리밍으로 들어오거나 한 번에 다 만들기 어려운 상황(외부 입력/온라인 처리)
- “가장 싼 간선부터 일부만” 처리하며 중간에 조기 종료가 확실한 특수 케이스

````


### 메두사와 전사들 

`````{admonition} Bottleneck constraints 관리 
:class: dropdown 

이 문제에서는 전사의 명수가 메모리/시간 차원에서 가장 큰 bottleneck이 된다. 따라서, 이러한 전사들의 위치와 생사문제를 관리하는 것이 중요해진다. 따라서, 해당 warriors들은 class Warrior로 전사마다 하나씩 만드는 행위는 매우 위험하다. 클래스 자체는 메모리를 많이 차지하기 때문에, 이것보다는 list[idx]차원으로 관리해주는 것이 더 유리하다. 

> war_ys: list[int] # war_ys[war_id] = y <br>
> war_xs: list[int] # war_xs[war_id] = x<br>
> alive: list[bool] # alive[war_id] = True <br>
> warriors_graph: dict[tuple(int, int), list[int]] defaultdict(list) # warriors_graph[(y, x)] = [war_id1, war_id2, ...]<br>

`````

````{admonition} 메두사의 시야각 
:class: dropdown 

특정 전사에 의해 메두사에게 가려지는 범위는 메두사와 해당 전사의 상대적인 위치에 의해 결정된다. 상하좌우 대각선 8방향을 나누었을 때, 메두사로부터 8방향 중 한 방향에 전사가 위치해있다면, 그 전사가 "동일한 방향으로 바라본 범위"에 포함된 모든 칸은 메두사에게 보이지 않습니다. 

![](../../assets/img/DFS_BFSPS/14.png)

또한, 메두사의 전체 시야각을 계산할 때, 아래와 같은 상황이 있으므로, 순차적으로 메두사의 전체 시야각을 1로 지정한뒤, 전사들의 시야각을 바라보는 "방향"에 맞춰 0으로 제거해주는 것이 포인트가 된다. 

![](../../assets/img/DFS_BFSPS/15.png)
````

````{admonition} Solution 
:class: dropdown 

```{literalinclude} ../solutions/DFS_BFSPS/4.py
:language: python 
```
````

````{admonition} test case 
:class: dropdown 

```{literalinclude} ../solutions/DFS_BFSPS/4_tc1.txt
:language: text
```

```{literalinclude} ../solutions/DFS_BFSPS/4_tc2.txt
:language: text
```

```{literalinclude} ../solutions/DFS_BFSPS/4_tc3.txt
:language: text
```
````

### 미지의 공간 탈출 

````{admonition} Idea 
:class: dropdown 

```{literalinclude} ../solutions/DFS_BFSPS/5.md
:language: md 
```
````

````{admonition} Solution 
:class: dropdown 


```{literalinclude} ../solutions/DFS_BFSPS/5.py
:language: python
:caption: 미지의 공간 탈출 Solution Code
```
````
````{admonition} test case 
:class: dropdown 

```{literalinclude} ../solutions/DFS_BFSPS/5_tc1.txt
:language: text 
```

```{literalinclude} ../solutions/DFS_BFSPS/5_tc2.txt
:language: text 
```
````
### 마법의 숲 탐색

```{admonition} 문제 정리
:class: dropdown 

1. 격좌/좌표: RxC를 HXC로 변환 (O, R-1) -> (0, H-1)까지 
2. move_golem(): 아래로 최대한 내려갈 수 있는 (move as far south as possible) 함수 구현 
   1. Step 2-1: rolling downward -> collision check 3 cells  
   2. Step 2-2: rotating left while moving downward -> collision check 5 cells 
   3. Step 2-3: rotating right while moving downward -> collision check 5 cells 
3. settle_or_reset():
   1. 정착 실패: 골렘이 멈췄을 때 십자 5칸 중 한 칸이라도 숲 밖(상단 패딩 포함 관점)이라면, 지금까지 놓인 모든 골렘을 전부 지우고 이번 시도는 0점 처리 후 다음 시도로 넘어간다.
   2. 정착 성공 시 배치: 정지 위치에 중심+팔 4칸을 기록하고, 출구 방향도 함께 저장한다(다음 단계 탐색에 필요). 
4. spirits 탐색 (explore)
   1. 골렘의 출구와 가까운 골렘을 통해 seed max값을 초기화 -> 8가지 가능한 방향으로 같은 component의 값들에 큰 값 -> 작은 값으로 value propagation(BFS사용)
   2. 관련하여 [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](
https://colab.research.google.com/github/Daye-Lee18/mybook/blob/main/assets/ipynb/propagation_demo.ipynb
) 시드의 최대값 초기화가 중요한 이유 확인 
5. solve(): 위의 함수들을 사용하여, input을 받고 결과 출력하는 함수 
   1. 격자/좌표: 
     - 숲은 R×C 격자. 문제 기준으로 가장 위가 1행, 가장 아래가 R행(좌표는 1-based). 정령(골렘)은 북쪽(격자 위쪽) 바깥에서 진입한다. 
       - (구현 팁) 진입·회전 동작을 편하게 처리하려고 상단에 3행을 더 붙인 격자(R+3×C) 로 두고 시뮬레이션하는 전형적인 방식이 많이 쓰인다. 
     - 입력 & 골렘 정의 (총 K번)
       - 한 번의 시도마다 열 c(1…C), 출구 방향 d(0:북, 1:동, 2:남, 3:서) 가 주어진다. 골렘은 십자(+) 모양(중심+상하좌우) 으로 5칸을 차지하며, 출구는 중심에서 d 방향으로 인접한 1칸이다. 
```


```{admonition} graph의 확장 
:class: dropdown
![3-1](../../assets/img/implementationPS/3/1.png)

In the beginning, the golem may extend up to `3 cells` above the forest. To handle this, we add 3 buffer rows at the top of the grid. These buffer rows are not part of the forest but provide enough room for the golem to move and settle safely. 
```
````{admonition} max value of each id
:class: dropdown
graph를 확장했으므로 (H= R + 3), best는 현재 골렘의 center_y, center_x보다 한 칸 아래인 것 ((cy +1, cx)이 맞으나, 
확장된 그래프 전의 index를 사용해야하므로 (cy + 1 -3, cx)이다. 하지만, 1열을 0이 아닌 1로 표시하므로 (cy + 1 -3 +1, cx)로 결국 (cy -1, cx)가 된다. 
````
```{admonition} 각 step 구현 
:class: dropdown

When the golem moves one step down, we must check 3 positions below the center: directly underneath, and one cell to the left and right. If all 3 are empty, the golem can move downward without collision. 
![3-2](../../assets/img/implementationPS/3/2.png)

Now let’s look at rotation to the left. For the golem to rotate, `five specific cells` must be empty. Interestingly, the rules allow rotation even if the upper-right or lower-right cells are occupied. We number the directions North, East, South, and West as 0, 1, 2, and 3. Using this, left rotation is simply (d + 3) mod 4, and right rotation is (d + 1) mod 4.

![3-3](../../assets/img/implementationPS/3/3.png)


Rotation to the right works symmetrically. Again, five surrounding cells must be empty to allow the move. The concept is the same as left rotation, but we apply the clockwise formula.
![3-4](../../assets/img/implementationPS/3/4.png)


When the golem finally stops, all four arms must be inside the forest area, which is from row 3 down to H–1. To manage state, we store each golem’s position and exit direction in units[gid]. We also record which exit belongs to which golem in exit_map. This way, exit_map tells us the gid, and units gives us the exact center and direction.
![3-5](../../assets/img/implementationPS/3/5.png)
```

```{admonition} 정보 저장 
:class: dropdown

We maintain two separate grids: `golem_arr` and `exit_map`. The `golem array` records body occupancy and is used for `collision checks`. The `exit map` records `only exits`, which lets us trace connectivity between golems using `BFS`. This separation is crucial because body cells and exit cells need to be treated differently.

![3-6](../../assets/img/implementationPS/3/6.png)
```

```{admonition} 최댓값 전파 BFS 
:class: dropdown

현재 골렘의 최댓값을 전파하려면, "다른 골렘의 '출구'"와 맞닿아있어야함. 따라서, 현재 골렘 8방에 다른 골렘의 출구가 있는지 출구 일때에만 전파를 해주는 것을 명심! 

![3-7](../../assets/img/implementationPS/3/7.png)

Finally, let’s look at how we propagate the maximum reachable row. 
- Step one: if `the current golem’s exit` touches a neighbor’s body, we inherit its `max_row` as the `seed` value. 
- Step two: using BFS, we spread values along exit-to-exit connections across the whole component. 

This two-step design ensures accuracy — newly placed golems immediately get the correct value. It also improves efficiency, since we avoid repeated updates by maximizing early and propagating just once.

![3-8](../../assets/img/implementationPS/3/8.png)

1. best 계산 단계 (시드 확정)
   1. `start_id` 의 출구 주변을 보면서, 이미 숲에 있던 이웃 골렘들이 가지고 있던 `max_row`값 참고 
   2. 방금 들어온 골렘이 가질 수 있는 가장 큰 시작값 (best)를 정함. 
2. 전파 단계 (BFS)
   1. 이제 `start_id`가 속한 component전체에 대해, max_row가 작은 이웃 노드들을 best 값으로 끌어올림. 
   2. 이 과정을 통해 같은 컴포넌트 안의 모든 골렘이 최대값을 공유 
   3. 이렇게 해 두면, 나중에 컴포넌트 안에서 어느 골렘을 시작점으로 잡아도 같은 답을 얻을 수 있습니다.
   4. 다음 골렘을 위해 준비 완료 

```

````{admonition} Solution 
:class: dropdown 

```{literalinclude} ../solutions/DFS_BFSPS/6.py
:language: python 
```
````

### 블록 이동하기  

````{admonition} 갈 수 있는 방향 설정 
:class: dropdown 

아래처럼 길이 나있는 경우가 있을 수 있으므로, 직선으로 움직이는 경우 상하좌우, 회전도 robot의 pivot block 기준으로 가로로 위치한 경우에는 위아래 혹은 세로로 위치한 경우에는 상하로 회전할 수 있도록 해야한다. 

![](../../assets/img/DFS_BFSPS/0.png)
````

````{admonition} 물체가 2개의 셀 이상 차지하는 경우, 효율적 visited 정보 저장 및 normalization
:class: dropdown

물체가 차지하는 셀이 1개가 아닌 두개 이기때문에 아래 2가지를 고려해야한다. 
- **Visited**: 물체 (Robot)이 차지하는 셀이 한 개 초과 즉, 이경우에는 board에 표시하면 memory가 초과되기 때문에 visited={} set으로 방문 여부를 체크해주면 좋다. 
- **Normalization**: 상태 정규화(순서 고정)도 존재해야한다. 즉, (y,x,t,v)와 (t,v,y,x)는 같은 로봇 상태인데, visited가 다르게 취급해 중복 상태 폭증하며 시간도 초과된다. 매번 (a,b) 두 좌표를 정렬해서 (small,big)로 저장하거나, frozenset({pos1,pos2})로 관리해야 한다. 즉, 정규화를 통해 (P1, P2) 중 작은 것이 앞에 오도록하여 같은 위치에 있는 로봇의 상태 체크를 잘 할 수 있게 된다. 

![](../../assets/img/DFS_BFSPS/1.png)

When we run BFS, each robot state is represented as the positions of its two blocks and the current time. But we need consistency: which block should be stored first? To avoid duplicates, we always order the two coordinates so that the smaller one comes first. This normalization guarantees that the same robot configuration is stored uniquely in the queue.
````

````{admonition} 물체의 rotation 및 예상 결과 확인
:class: dropdown

If the robot is lying horizontally, we can rotate it around either the left block or the right block. Each rotation can go both upward and downward, converting the robot into a vertical orientation. So, in total, we get four possible rotations in this situation.

![](../../assets/img/DFS_BFSPS/2.png)

During rotation, we must check not only the pivot block but also the adjacent cells that the robot sweeps through.
- If the pivot is the right block, the non-pivot’s upper and lower cells must be empty.
- If the pivot is the left block, again the non-pivot’s upper and lower cells must also be empty.
These checks prevent collisions during rotation.


![](../../assets/img/DFS_BFSPS/3.png)

After rotation, the final state is defined by the pivot block plus the new block either above or below it.
For example, rotating upward results in the pivot plus the cell above it.
Rotating downward results in the pivot plus the cell below it.
This ensures that we represent the robot’s new vertical position consistently.

![](../../assets/img/DFS_BFSPS/4.png)

```{code-block} python
# [Left pivot rotation ↑]
#   Pivot       → (y1, x1)
#   After move  → {(y1, x1), (y1-1, x1)}
#   Must check  → (y2-1, x2)

# [Right pivot rotation ↑]
#   Pivot       → (y2, x2)
#   After move  → {(y2-1, x2), (y2, x2)}
#   Must check  → (y1-1, x1)
```

After rotation, the final state is defined by the pivot block plus the new block either above or below it. For example, rotating upward results in the pivot plus the cell above it.
Rotating downward results in the pivot plus the cell below it.
This ensures that we represent the robot’s new vertical position consistently.


![](../../assets/img/DFS_BFSPS/5.png)


When the robot is vertical, the situation is symmetric.
The pivot can be either the top block or the bottom block.
Each pivot allows a rotation to the left or to the right, changing the robot’s orientation from vertical to horizontal.

![](../../assets/img/DFS_BFSPS/6.png)

As in the horizontal case, rotation requires collision checks.
If the pivot is the bottom block and we rotate left, the non-pivot’s left cell must be empty.
If we rotate right, the non-pivot’s right cell must be empty.
Similarly, when the pivot is the top block, we check the left and right cells of the non-pivot during rotation.
These rules guarantee that rotations happen without intersecting obstacles.

![](../../assets/img/DFS_BFSPS/7.png)

After a vertical rotation, the final state is also described by the pivot plus one adjacent cell.
Rotating left results in the pivot plus its left neighbor.
Rotating right results in the pivot plus its right neighbor.
This completes the transition from vertical to horizontal while preserving a consistent representation.
````

````{admonition} Complexity
:class: dropdown
By exploring the state space, we obtain a complexity of $O(N^2)$.
The grid size is $N \times N$, and the robot can place one of its ends on any cell. This gives $O(N^2)$ possibilities. Since the robot can exist in two orientations—horizontal and vertical—each cell has two possible states. Therefore, the total number of states is approximately $O(2 \times N^2)$, which simplifies to $O(N^2)$.

In addition, the number of possible actions from each state is constant: 8 moves in total (4 parallel moves in the four directions, plus 4 rotations — 2 pivots × 2 rotation directions). Thus, each state expands in $O(1)$.

Consequently, the overall time complexity is $O(N^2)$, and the space complexity is also $O(N^2)$.
````


```{toggle}
```{code-block} python
---
caption: "Solution" 
---
"""
제한사항 
5 <= N <= 100 
BFS -> O(NxN), 각 칸에 대해 2가지 방향, 각 상태에서 상수개 액션 
# 장애물이 있으면 왼쪽/위쪽으로 돌아가야하므로 4방향 모두 탐색해야함. 
# 회전도 양방향 회전 전부 고려해야함. 
"""
from collections import deque 

def solution(board):
    n = len(board)

    def in_range(y, x):
        return 0 <= y < n and 0 <= x < n

    def neighbors(p1, p2):
        (y1, x1), (y2, x2) = p1, p2
        cand = []

        # 1) 4방향 평이동
        dirs = [(-1,0),(1,0),(0,-1),(0,1)]
        for dy, dx in dirs:
            ny1, nx1 = y1 + dy, x1 + dx
            ny2, nx2 = y2 + dy, x2 + dx
            if in_range(ny1, nx1) and in_range(ny2, nx2) \
               and board[ny1][nx1] == 0 and board[ny2][nx2] == 0:
                cand.append(((ny1, nx1), (ny2, nx2)))

        # 2) 회전 (가로 ↔ 세로)
        if y1 == y2:  # 가로일 때 → 세로로 회전
            for d in [-1, 1]:  # 위/아래
                if in_range(y1 + d, x1) and in_range(y2 + d, x2) \
                   and board[y1 + d][x1] == 0 and board[y2 + d][x2] == 0:
                    # 왼쪽 블록 기준 회전
                    cand.append(((y1, x1), (y1 + d, x1)))
                    # 오른쪽 블록 기준 회전
                    cand.append(((y2, x2), (y2 + d, x2)))
        elif x1 == x2:  # 세로일 때 → 가로로 회전
            for d in [-1, 1]:  # 좌/우
                if in_range(y1, x1 + d) and in_range(y2, x2 + d) \
                   and board[y1][x1 + d] == 0 and board[y2][x2 + d] == 0:
                    # 위쪽 블록 기준 회전
                    cand.append(((y1, x1), (y1, x1 + d)))
                    # 아래쪽 블록 기준 회전
                    cand.append(((y2, x2), (y2, x2 + d)))

        # 상태 정규화 (작은 좌표가 앞으로)
        norm = []
        for a, b in cand:
            if a <= b:
                norm.append((a, b))
            else:
                norm.append((b, a))
        return norm

    # 3) BFS 시작
    start = ((0, 0), (0, 1))  # 시작 상태
    q = deque([(start, 0)])
    visited = {start} # visited도 graph위에 체크하는 것이 아닌, Set으로 관리하여 메모리 효율적으로 관리 

    goal = (n - 1, n - 1)

    while q:
        (p1, p2), t = q.popleft()
        if p1 == goal or p2 == goal:
            return t
        for nxt in neighbors(p1, p2):
            if nxt not in visited:
                visited.add(nxt)
                q.append((nxt, t + 1))

    return -1  # 도달 불가


if __name__ == '__main__':
    board = [[0, 0, 0, 1, 1],
             [0, 0, 0, 1, 0],
             [0, 1, 0, 1, 1],
             [1, 1, 0, 0, 1],
             [0, 0, 0, 0, 0]]  # Expected 7
    print(solution(board))

    board = [[0, 0], 
             [0, 0]]  # Expected 1
    print(solution(board))

    board = [[0, 0, 0], 
             [0, 0, 0], 
             [0, 0, 0]]  # Expected 3
    print(solution(board))

```

### 고대 문명 유적 탐사 

```{admonition} 리스트의 복사
:class: dropdown
### 리스트의 복사 

- 1차원 리스트 같은 경우에는 `[:]`로 복사하면 된다. 

- 2차원 리스트는 [:]로 복사하면 얕은 복사라서, 내부 행 리스트를 공유하게 된다. 따라서 `from copy import deepcopy`를 사용하거나 `list comprehension`를 사용하면 된다. 예를 들어 다음 코드와 같이 할 수 있다. 

```{code-block} python
grid = [[1, 2], [3, 4]]

# 깊은 복사
copied = [row[:] for row in grid]

copied[0][0] = 99
print(grid)   # [[1, 2], [3, 4]]
print(copied) # [[99, 2], [3, 4]]
```



````{admonition} Rotation around a center 
:class: dropdown 
 
이번 문제에서는 center가 변화하면서 rotation을 수행해야하기 때문에, 해당 center에서 3x3 rotation을 직접 종이에 적어본 후, 행열이 각각 어떻게 변하는지 `예상 결과`와 동일한지 하나하나 따져가며 구현해야한다. 말그대로 구현 문제! 

즉, 5x5 행렬에서는 총 9개의 센터가 존재하고, 각 센터마다 90 -> 180 -> 270 회전 (27번) 중에서 가장 우선순위가 높은 것만 저장하면 된다. 
또한, 각 센터에서 90 회전 다음에 또 90회전을 하면 180도이므로 CW 90도 회전 한 번을 구현해놓으면 코드가 간단해진다. 

center가 (1,1)일때 CW 90도 회전 이후 결과 
![](../../assets/img/DFS_BFSPS/8.png) <br>
center가 (1,2)일때 CW 90도 회전 이후 결과 
![](../../assets/img/DFS_BFSPS/9.png) <br>
center가 (1,3)일때 CW 90도 회전 이후 결과 
![](../../assets/img/DFS_BFSPS/10.png) <br>
center가 (2,1)일때 CW 90도 회전 이후 결과 
![](../../assets/img/DFS_BFSPS/11.png) <br>
````

````{admonition} rotation.py degugging file 
:class: dropdown 

아래 파일을 따로 만들어서, 실제 로테이션이 잘 되는지 확인하였다. 문제를 풀 때 solve()함수는 다양한 함수들로 이루어져있어, 디버깅이 복잡하다. 따라서, 하나의 함수마다 degugging은 이런 식으로 별도로 output을 출력해서 확인하면 쉬워진다. 

```{code-block} python
:caption: rotation.py (debug)
def in_circle(y, x, cy, cx):
    return cy -1 <= y <= cy + 1 and cx -1 <= x <= cx + 1

def my_function(input) -> int:
    N = len(input)
    centers = [(1, 1), (1, 2), (1, 3), (2,1), (2,2), (2, 3), (3, 1), (3, 2), (3, 3)]

    for cy, cx in centers:
        local_graph = [row[:] for row in graph]
        previous_local_graph = [row[:] for row in local_graph]

        print(f"Original Graph")
        for row in local_graph:
            print(row)

        for rotation_cnt in range(3):
            for y in range(N):
                for x in range(N):
                    if not (y==cy and x==cx) and in_circle(y, x, cy, cx):
                        add_num = cy + cx 
                        x_minus_y = cx - cy 
                        local_graph[y][x] = previous_local_graph[add_num-x][y + x_minus_y] # CW 90도 회전 
                    else:
                        local_graph[y][x] = previous_local_graph[y][x]
            print(f"Center {cy, cx}, CW {90*(rotation_cnt+1)} degree : ")
            for row in local_graph:
                print(row)

            previous_local_graph = [row[:] for row in local_graph]
    return local_graph

if __name__ == '__main__':
    global graph 
    graph = [
        [1, 2, 3, 10, 11],
        [4, 5, 6, 12, 13],
        [7, 8, 9, 14, 15],
        [16, 17, 18, 19, 20],
        [20, 21, 22, 23, 24],
    ]

    result = my_function(graph)
```
```{code-block} text
---
caption: Output of rotation.py 
---
    Original Graph
    [1, 2, 3, 10, 11]
    [4, 5, 6, 12, 13]
    [7, 8, 9, 14, 15]
    [16, 17, 18, 19, 20]
    [20, 21, 22, 23, 24]
    Center (1, 1), CW 90 degree : 
    [7, 4, 1, 10, 11]
    [8, 5, 2, 12, 13]
    [9, 6, 3, 14, 15]
    [16, 17, 18, 19, 20]
    [20, 21, 22, 23, 24]
    Center (1, 1), CW 180 degree : 
    [9, 8, 7, 10, 11]
    [6, 5, 4, 12, 13]
    [3, 2, 1, 14, 15]
    [16, 17, 18, 19, 20]
    [20, 21, 22, 23, 24]
    Center (1, 1), CW 270 degree : 
    [3, 6, 9, 10, 11]
    [2, 5, 8, 12, 13]
    [1, 4, 7, 14, 15]
    [16, 17, 18, 19, 20]
    [20, 21, 22, 23, 24]
    Original Graph
    [1, 2, 3, 10, 11]
    [4, 5, 6, 12, 13]
    [7, 8, 9, 14, 15]
    [16, 17, 18, 19, 20]
    [20, 21, 22, 23, 24]
    Center (1, 2), CW 90 degree : 
    [1, 8, 5, 2, 11]
    [4, 9, 6, 3, 13]
    [7, 14, 12, 10, 15]
    [16, 17, 18, 19, 20]
    [20, 21, 22, 23, 24]
    Center (1, 2), CW 180 degree : 
    [1, 14, 9, 8, 11]
    [4, 12, 6, 5, 13]
    [7, 10, 3, 2, 15]
    [16, 17, 18, 19, 20]
    [20, 21, 22, 23, 24]
    Center (1, 2), CW 270 degree : 
    [1, 10, 12, 14, 11]
    [4, 3, 6, 9, 13]
    [7, 2, 5, 8, 15]
    [16, 17, 18, 19, 20]
    [20, 21, 22, 23, 24]
    Original Graph
    [1, 2, 3, 10, 11]
    [4, 5, 6, 12, 13]
    [7, 8, 9, 14, 15]
    [16, 17, 18, 19, 20]
    [20, 21, 22, 23, 24]
    Center (1, 3), CW 90 degree : 
    [1, 2, 9, 6, 3]
    [4, 5, 14, 12, 10]
    [7, 8, 15, 13, 11]
    [16, 17, 18, 19, 20]
    [20, 21, 22, 23, 24]
    Center (1, 3), CW 180 degree : 
    [1, 2, 15, 14, 9]
    [4, 5, 13, 12, 6]
    [7, 8, 11, 10, 3]
    [16, 17, 18, 19, 20]
    [20, 21, 22, 23, 24]
    Center (1, 3), CW 270 degree : 
    [1, 2, 11, 13, 15]
    [4, 5, 10, 12, 14]
    [7, 8, 3, 6, 9]
    [16, 17, 18, 19, 20]
    [20, 21, 22, 23, 24]
    Original Graph
    [1, 2, 3, 10, 11]
    [4, 5, 6, 12, 13]
    [7, 8, 9, 14, 15]
    [16, 17, 18, 19, 20]
    [20, 21, 22, 23, 24]
    Center (2, 1), CW 90 degree : 
    [1, 2, 3, 10, 11]
    [16, 7, 4, 12, 13]
    [17, 8, 5, 14, 15]
    [18, 9, 6, 19, 20]
    [20, 21, 22, 23, 24]
    Center (2, 1), CW 180 degree : 
    [1, 2, 3, 10, 11]
    [18, 17, 16, 12, 13]
    [9, 8, 7, 14, 15]
    [6, 5, 4, 19, 20]
    [20, 21, 22, 23, 24]
    Center (2, 1), CW 270 degree : 
    [1, 2, 3, 10, 11]
    [6, 9, 18, 12, 13]
    [5, 8, 17, 14, 15]
    [4, 7, 16, 19, 20]
    [20, 21, 22, 23, 24]
    Original Graph
    [1, 2, 3, 10, 11]
    [4, 5, 6, 12, 13]
    [7, 8, 9, 14, 15]
    [16, 17, 18, 19, 20]
    [20, 21, 22, 23, 24]
    Center (2, 2), CW 90 degree : 
    [1, 2, 3, 10, 11]
    [4, 17, 8, 5, 13]
    [7, 18, 9, 6, 15]
    [16, 19, 14, 12, 20]
    [20, 21, 22, 23, 24]
    Center (2, 2), CW 180 degree : 
    [1, 2, 3, 10, 11]
    [4, 19, 18, 17, 13]
    [7, 14, 9, 8, 15]
    [16, 12, 6, 5, 20]
    [20, 21, 22, 23, 24]
    Center (2, 2), CW 270 degree : 
    [1, 2, 3, 10, 11]
    [4, 12, 14, 19, 13]
    [7, 6, 9, 18, 15]
    [16, 5, 8, 17, 20]
    [20, 21, 22, 23, 24]
    Original Graph
    [1, 2, 3, 10, 11]
    [4, 5, 6, 12, 13]
    [7, 8, 9, 14, 15]
    [16, 17, 18, 19, 20]
    [20, 21, 22, 23, 24]
    Center (2, 3), CW 90 degree : 
    [1, 2, 3, 10, 11]
    [4, 5, 18, 9, 6]
    [7, 8, 19, 14, 12]
    [16, 17, 20, 15, 13]
    [20, 21, 22, 23, 24]
    Center (2, 3), CW 180 degree : 
    [1, 2, 3, 10, 11]
    [4, 5, 20, 19, 18]
    [7, 8, 15, 14, 9]
    [16, 17, 13, 12, 6]
    [20, 21, 22, 23, 24]
    Center (2, 3), CW 270 degree : 
    [1, 2, 3, 10, 11]
    [4, 5, 13, 15, 20]
    [7, 8, 12, 14, 19]
    [16, 17, 6, 9, 18]
    [20, 21, 22, 23, 24]
    Original Graph
    [1, 2, 3, 10, 11]
    [4, 5, 6, 12, 13]
    [7, 8, 9, 14, 15]
    [16, 17, 18, 19, 20]
    [20, 21, 22, 23, 24]
    Center (3, 1), CW 90 degree : 
    [1, 2, 3, 10, 11]
    [4, 5, 6, 12, 13]
    [20, 16, 7, 14, 15]
    [21, 17, 8, 19, 20]
    [22, 18, 9, 23, 24]
    Center (3, 1), CW 180 degree : 
    [1, 2, 3, 10, 11]
    [4, 5, 6, 12, 13]
    [22, 21, 20, 14, 15]
    [18, 17, 16, 19, 20]
    [9, 8, 7, 23, 24]
    Center (3, 1), CW 270 degree : 
    [1, 2, 3, 10, 11]
    [4, 5, 6, 12, 13]
    [9, 18, 22, 14, 15]
    [8, 17, 21, 19, 20]
    [7, 16, 20, 23, 24]
    Original Graph
    [1, 2, 3, 10, 11]
    [4, 5, 6, 12, 13]
    [7, 8, 9, 14, 15]
    [16, 17, 18, 19, 20]
    [20, 21, 22, 23, 24]
    Center (3, 2), CW 90 degree : 
    [1, 2, 3, 10, 11]
    [4, 5, 6, 12, 13]
    [7, 21, 17, 8, 15]
    [16, 22, 18, 9, 20]
    [20, 23, 19, 14, 24]
    Center (3, 2), CW 180 degree : 
    [1, 2, 3, 10, 11]
    [4, 5, 6, 12, 13]
    [7, 23, 22, 21, 15]
    [16, 19, 18, 17, 20]
    [20, 14, 9, 8, 24]
    Center (3, 2), CW 270 degree : 
    [1, 2, 3, 10, 11]
    [4, 5, 6, 12, 13]
    [7, 14, 19, 23, 15]
    [16, 9, 18, 22, 20]
    [20, 8, 17, 21, 24]
    Original Graph
    [1, 2, 3, 10, 11]
    [4, 5, 6, 12, 13]
    [7, 8, 9, 14, 15]
    [16, 17, 18, 19, 20]
    [20, 21, 22, 23, 24]
    Center (3, 3), CW 90 degree : 
    [1, 2, 3, 10, 11]
    [4, 5, 6, 12, 13]
    [7, 8, 22, 18, 9]
    [16, 17, 23, 19, 14]
    [20, 21, 24, 20, 15]
    Center (3, 3), CW 180 degree : 
    [1, 2, 3, 10, 11]
    [4, 5, 6, 12, 13]
    [7, 8, 24, 23, 22]
    [16, 17, 20, 19, 18]
    [20, 21, 15, 14, 9]
    Center (3, 3), CW 270 degree : 
    [1, 2, 3, 10, 11]
    [4, 5, 6, 12, 13]
    [7, 8, 15, 20, 24]
    [16, 17, 14, 19, 23]
    [20, 21, 9, 18, 22]
```
````

````{toggle}
```{code-block} python
# 5x5중에서 3x3격자 선택 및 회전 CW: 90도, 180도, 270도 
# 향상 회전 : 중심좌표를 기준으로 90도 회전 
'''
### 탐사 진행: 회전 목표 
1)유물 1차 회득 가치 최대화
2) 1)의 방법이 여러개이면, 회전한 각도 중 가장 작은 각도 선택
3) 2)의 방법도 여러가지이면 (회전 중심좌표가 다를 수 있음), 회전 중심 좌표의 열이 가장 작은 구간 선택 
4) 열이 같다면 행이 가장 작은 구간 선택 

### 유물 1차 획득 
- 유물의 가치: 5x5행렬에서 모인 조각의 개수 -> "3개 이상"부터 획득 가능  
- 유물이 사라짐. 
- 새로들어오는 유물은 유적의 벽면에 써 있는 숫자대로 진행 (row up, column up순으로 채워짐)
- 사용된 숫자다음부터 다음에 사용할 수 있음 

#### 유물 연쇄 획득 
- 새로운 유물 조각이 생겨난 이후에도 유물이 있으면 조각을 획득하고 없앤후 다시 채움.
- 다만 더 이상 조각이 3개 이상 연결되지 않아 유물이 될 수 없으면 멈춤 

#### 탐사 반복 
- 탐사 진행 -> 유물 1차 획득 -> 유믈 연쇄 획득 과정까지 1턴이며 총 K번 턴을 돌림. 
- 1번의 turn에서 K번 이전에 어떠한 방법을 사용해서라도 유물을 획득할 수 없다면, 모든 탐사는 그 즉시 종료됨. 
이 경우 얻을 수 있는 유물이 존재하지 않으므로, 종료되는 턴에 아무 값도 출력하지 않음. 
'''

from collections import deque 
from typing import List 

def solve():

    # f = open('/Users/dayelee/Documents/GitHub/mybook/Input.txt', 'r')
    K, M = map(int, input().split())
    global graph, parts
    graph = []

    for n in range(5):
        graph.append(list(map(int, input().split())))

    # 유물조각은 들어온 순서부터 pop
    parts = deque(list(map(int, input().split())))

    
    for k in range(K):
        total = 0

        # Step 1: 
        # 3x3을 회전: 총 9개 위치를 중심으로 CW 90, 180, 270도 (9 * 3=27)개 중 선택, 유물은 조각 3개 이상 연결 
        # 유적위치 locs = list(), 유물의 가치 = len(locs), 
        locs, result_graph = explore()

        if len(locs) == 0:
            break # 유적의 가치가 없으면 K 턴 전에 stop 
        
        total += len(locs) # 유물의 가치 더하기 
        graph = result_graph[:] # 유적 graph update 

        # global graph에 유적 위치 Locs에 새로운 조각 update 
        update_graph(locs)
        
        # global graph에 유물 연쇄 획득 과정 
        value= get_chained_parts()
        total += value 

        # 공백을 사이에 두고 출력 
        print(total, end=' ')


def compare(fy, fx, ry, rx):
    if fx != rx: 
        return fx < rx   # 열 번호가 작은 순 
    elif fy != ry:
        return fy > ry  # 행 번호가 큰 순 
    return True 

def sort_locs(locs: List):
    '''
    ascending order 
    '''
    N = len(locs)
    for f_pointer in range(N):
        lowest_pointer = f_pointer 
        for r_pointer in range(f_pointer+1, N):
            if not compare(locs[lowest_pointer][0], locs[lowest_pointer][1], locs[r_pointer][0], locs[r_pointer][1]):
                # 저장 
                lowest_pointer = r_pointer 
        # swap 
        if lowest_pointer != f_pointer:
            temp = locs[lowest_pointer]
            locs[lowest_pointer] = locs[f_pointer]
            locs[f_pointer] = temp 


def update_graph(locs):
    global graph, parts 

    sort_locs(locs) # call by reference 
    # 정렬 순서대로 update 
    for cy, cx in locs:
        graph[cy][cx] = parts.popleft()



def in_circle(y, x, cy, cx):
    return cy -1 <= y <= cy + 1 and cx -1 <= x <= cx + 1

def explore():
    global graph
    # 열이 가장 작고 -> 행이 가장 작은 순으로 배열  
    # centers = [(1, 1), (1, 2), (1, 3), (2,1), (2,2), (2, 3), (3, 1), (3, 2), (3, 3)]
    centers = [(1, 1), (2, 1), (3, 1), (1, 2), (2, 2), (3, 2), (1, 3), (2, 3), (3, 3)]

    max_value = 0
    max_locs = []
    min_rotation = 99999999 
    result_graph =[[0] * 5 for _ in range(5)]
    for cy, cx in centers:
        local_graph = [row[:] for row in graph]
        previous_local_graph = [row[:] for row in local_graph]
        # 90, 180, 270 CW rotation 
        for rotation_cnt in range(3):
            # 회전 후의 새로운 graph생성 
            add_num = cy + cx 
            x_minus_y = cx - cy 
            for y in range(5):
                for x in range(5):
                    # if (y!=cy and x!=cx) and in_circle(y, x, cy, cx):
                    if not (y==cy and x==cx) and in_circle(y, x, cy, cx):
                        local_graph[y][x] = previous_local_graph[add_num-x][y+x_minus_y] # CW 90도 회전 
                    else:
                        local_graph[y][x] = previous_local_graph[y][x]

            # 이전 rotaed graph update 
            previous_local_graph = [row[:] for row in local_graph]

            # rotation 후 고정된 Graph에서 3개 이상 모여있는 유물의 위치 계산
            cur_locs = calculate_values(previous_local_graph)
            
            # locs, rotation_cnt비교 
            # 각도가 작은 것이 제일 먼저 priority : 각도가 같으면, 열 -> 행 순서대로 이미 적용되어있기 때문에, 가치가 더 클때만 바꾼다. 
            # 따라서 오직 이전것보다 큰 경우에만 update (같으면 앞의 것으로 함.)
            if len(cur_locs) >= max_value: # Step 1: 유물 가치가 가장 높은 것을 최대화 
                if len(cur_locs) == max_value: # Step 2: Step 1이 여러개라면, 회전 각도가 가장 작은 것 
                    if min_rotation > rotation_cnt:
                        # update 
                        min_rotation = rotation_cnt 
                        max_value = len(cur_locs)
                        max_locs = list(cur_locs)
                        result_graph = [row[:] for row in previous_local_graph]
                else:
                    # update 
                    min_rotation = rotation_cnt 
                    max_value = len(cur_locs)
                    max_locs = list(cur_locs)
                    result_graph = [row[:] for row in previous_local_graph]

    return max_locs,result_graph

def in_range(y, x):
    return 0<=y<5 and 0<=x <5 

def BFS(y, x, cur_graph):
    q = deque([(y, x)])
    visited = set()
    visited.add((y, x))
    DY = [-1, 1, 0, 0]; DX = [0, 0, 1, -1]

    while q:
        cur_y, cur_x = q.popleft()

        for t in range(4):
            ny = cur_y + DY[t]; nx = cur_x + DX[t] 
            # range안에 있고 & 원래 y, x 안에 있는 수와 옆의 수가 동일한지 & 방문하지는 않았는지 
            if in_range(ny, nx) and cur_graph[ny][nx] == cur_graph[y][x] and (not (ny, nx) in visited):
                q.append((ny, nx))
                visited.add((ny,nx))

    # 3개 이상인지 
    if len(visited) >= 3:
        return True, visited
    else :
        return False, None 

def calculate_values(cur_graph):
    '''
    고정된 Graph에서 3개 이상 모여있는 유물의 위치 계산
    '''
    result = set()

    for y in range(5):
        for x in range(5):
            is_more_three, locs = BFS(y, x, cur_graph)
            if is_more_three:
                # set을 extend하는 방법? 
                result = result.union(locs)
                    
    return result 
            

def get_chained_parts():
    global graph 
    values = 0 
    while True:
        # 유적 위치 세기 
        locs = calculate_values(graph)

        if len(locs) == 0:
            break 
 
        values += len(locs) 

        # 유적 graph 업데이트하기 
        update_graph(list(locs))
    return values


if __name__ == '__main__':
    solve()
```