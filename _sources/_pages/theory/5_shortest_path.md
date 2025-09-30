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

# Lecture 5-1. Shortest Path

최단 경로 (Shotest Path) 알고리즘은 가장 짧은 경로를 찾는 알고리즘이다. 그래서 "길 찾기" 문제라고도 불린다. 최단 경로 알고리즘 유형에는 다양한 종류가 있다. 예를 들어 '한 지점에서 다른 특정 지점까지의 최단 경로를 구해야 하는 경우", "모든 지점에서 다른 모든 지점까지의 최단 경로를 모두 구해야하는 경우" 등의 다양한 사례가 존재한다. 각 사례에 맞는 알고리즘을 알고 있다면 좀 더 쉽게 풀 수 있다. 이번 챕터에서는 **다익스트라 최단 경로**와 **플로이드 워셔 알고리즘** 유형을 다룰 것이다. 

사실 이번 장에서 다루는 내용은 그리디 알고리즘 및 다이나믹 프로그래밍 알고리즘의 한 유형으로 볼 수 있다. 

## Dijkstra Algorithm 

다익스트라 (Dijkstra) 최단 경로 알고리즘은 그래프에서 여러 개의 노드가 있을 때, **특정한 노드에서 출발**하여 **다른 노드로 가는 각각의 최단경로**를 구해주는 알고리즘이다(출발: 1 -> 도착: the rest of the nodes). 또한 다익스트라 최단 경로 알고리즘은 **'음의 간선'**이 없을 때 정상적으로 동작한다. 따라서, 현재 세계의 길에서 간선은 음의 간선으로 표현되지 않으므로 GPS 소프트웨어의 기본 알고리즘으로 채택되곤 한다. 

````{admonition} Dijkstra Algorithm Conditions
:class: important
1. One Start Node -> The Rest of The Nodes 
2. All Integer Edge Weight 
````

Dijkstra Algorithm는 다음과 같은 원리를 따른다. 

1) 출발 노드를 설정한다.
2) **최단 거리 테이블**을 초기화한다. 
3) 방문하지 않은 노드 중에서 최단 거리가 가장 짧은 노드를 선택한다. 
4) 해당 노드를 거쳐 다른 노드로 가는 비용을 계산하여 최단 거리 테이블을 갱신한다. 
5) 위 과정에서 3), 4)번 과정을 반복한다. 

위의 Step 2)에서 최단 거리 테이블은 "각 노드에 대한 현재까지의 최단 거리" 정보를 저장한다. 또한 매번 현재 처리하고 있는 노드를 기준으로 주변 간선을 확인한다. 나중에 현재 처리하고 있는 노드와 인접한 노드로 도달하는 더 짧은 경로를 찾으면 "더 짧은 경로도 있었네? 이제부터는 이 경로가 제일 짧은 경로야"라고 판단하는 것이다. 따라서 "방문하지 않은 노드 중에서 현재 최단 거리가 가장 짧은 노드를 확인"해 그 노드에 대하여 4) 과정을 수행한다는 점에서 그리디 알고리즘으로 볼 수 있다. 

다익스트라 알고리즘을 구현할 때 느리지만 구현하기 쉬운 방법이 있고 구현하기 까다롭지만 빠르게 동작하는 방법이 있는데, 코딩 시험을 위해서 두 번째 방법을 당연히 숙지하고 있어야한다. 

### Dijkstra Algorithm 예시 

다음과 같은 그래프가 있을 때, 1번 노드에서 다른 모든 노드로 가는 최단 경로를 구하는 문제를 생각해보자. 출발 노드는 1이라 가정한다. 즉, 1번 노드에서 다른 모든 노드로 가는 최단 거리를 계산할 것이다. 최단 경로 테이블을 보면, **"초기 상태에서는 다른 모든 노드로 가는 최단 거리"**를 "INF"로 초기화한다. 즉, 테이블의 각 값은 **"해당 노드로 도달하는 가장 짧은 값 "**을 저장한다. 파이썬에서 기본으로 1e9를 실수 자료형으로 처리하므로 모든 간선이 정수형으로 표현되는 문제에서는 `int(1e9)`로 초기화한다. (대부분의 문제에서는 그래프의 간신 길이 정보를 줄 때 1억 미만의 값으로 준다.)

![image](../../assets/img/shortest_path/1.png)


**Step 1** 

먼저 방문하지 않은 노드 중에서 최단 거리가 가장 짧은 노드를 선택하는데, "출발 노드에서 출발 노드"로의 거리는 0으로 보기 때문에, 처음에는 출발 노드가 선택된다. 

이제 1번 노드를 거쳐 다른 노드로 가는 비용을 계산한다. 1번 노드와 연결된 모든 간선을 하나씩 확인한다. 현재 1번 노드까지 오는 비용은 0이므로, 1번 노드를 거쳐 2번, 3번, 4번 노드로 가는 최소 비용은 차례대로 2(0+2), 5(0+5), 1(0+1)이다. 현재 2, 3, 4번 노드로 가는 비용이 '무한'이므로 세 노드에 도달하는 더 짧은 경로를 찾았으므로 각각 새로운 값으로 갱신한다. 

![2](../../assets/img/shortest_path/2.png)

**Step 2**

이후의 모든 단계에서도 마찬가지로 방문하지 않은 노드 중에서 최단 거리가 가장 짧은 노드를 선택한다. 따라서 4번 노드가 선택된다. 이어서 4번 노드를 거쳐서 갈 수 있는 노드를 확인한다. 4번 노드에서 갈 수 있는 노드는 3번과 5번이고, 4번 노드까지의 거리는 1이므로 4번 노드를 거쳐서 3번과 5번 노드로 가는 최소 비용은 차례대로 4(1+3), 2(1+1)이다. 이 두 값은 기존의 리스트에 담겨 있던 값보다 작으므로 다음처럼 리스트가 갱신된다. 

![3](../../assets/img/shortest_path/3.png)

**Step 3**

방문하지 않은 노드 중 최단 거리가 가장 짧은 노드 선택: 2번 
2번 노드를 거쳐 갈 수 있는 노드들 중 방문하지 않은 노드들: 3번, 4번 
2번 노드를 거쳐 가는 경우 각 노드들까지의 최종 비용 계산: 3번 (2+3 vs. 4 -> 갱신 x), 4번 (2+2 vs. 1 -> 갱신 x)

![4](../../assets/img/shortest_path/4.png)

**Step 4**

방문하지 않은 노드 중 최단 거리가 가장 짧은 노드 선택: 5번 
2번 노드를 거쳐 갈 수 있는 노드들 중 방문하지 않은 노드들: 3번, 6번 
2번 노드를 거쳐 가는 경우 각 노드들까지의 최종 비용 계산: 3번 (2+1 vs. 4 -> 갱신), 6번 (2+2 vs. INF -> 갱신)

![5](../../assets/img/shortest_path/5.png)

**Step 5**

방문하지 않은 노드 중 최단 거리가 가장 짧은 노드 선택: 3번 
2번 노드를 거쳐 갈 수 있는 노드들 중 방문하지 않은 노드들: 2번, 6번 
2번 노드를 거쳐 가는 경우 각 노드들까지의 최종 비용 계산: 2번 (3+3 vs. 4 -> 갱신 x), 6번 (3+5 vs. 4 -> 갱신 x)

![6](../../assets/img/shortest_path/6.png)

**Step 6**

방문하지 않은 노드 중 최단 거리가 가장 짧은 노드 선택: 6번 
2번 노드를 거쳐 갈 수 있는 노드들 중 방문하지 않은 노드들: x
2번 노드를 거쳐 가는 경우 각 노드들까지의 최종 비용 계산: x

![7](../../assets/img/shortest_path/7.png)

**Summary**

최단 거리 테이블이 의미하는 바는 1번 노드 **(시작 노드)**로부터 출발했을 때 2번, 3번, 4번, 5번, 6번 노드까지 가기 위한 최단 경로 (비용 최단 경로) 가 각각 2, 3, 1, 2, 4라는 의미이다. 

다익스트라 최단 경로 알고리즘에서는 '방문하지 않은 노드 중에서 가장 최단 거리가 짧은 노드를 선택'하는 과정을 반복하는데, 이렇게 선택된 노드는 '최단 거리'가 완전히 선택된 노드이므로, 더 이상 알고리즘을 반복해도 최단 거리가 줄어들지 않는다. 앞서 [Step 6]까지의 모든 경우를 확인해보면 실제로 한 번 선택된 노드는 최단 거리가 감소하지 않는다. 예를 들어, [step 2]에서는 4번 노드가 선택되어서 4번 노드를 거쳐서 이동할 수 있는 경로를 확인했다. 이후에 [step 3] ~ [step 6]이 진행되었지만, 4번 노드에 대한 최단 거리는 더 이상 감소하지 않았음을 확인할 수 있다. 다시 말해 다익스트라 알고리즘은 **한 단계당 하나의 노드에 대한 최단 거리를 확실히 찾는 것**으로 이해할 수 있다. 즉, 이미 방문한 노드는 이후에도 테이블 값이 바뀌지 않는다. 

```{admonition} Dijkstra Summary
:class: important 

**개념**  
- 하나의 시작 노드에서 출발하여, *아직 방문하지 않은 노드 중 최단 거리 추정값이 가장 작은 노드*를 선택해 거리 테이블을 갱신.  
- 우선순위 큐(min-heap)에서 뽑힌 노드는 그 순간 최단 경로가 확정되며, 이후로는 갱신되지 않음.  

**구현 절차**  
1. 최단 거리 테이블을 `INF`로 초기화, 시작 노드는 0으로 설정.  
2. `heapq`에 `(거리, 노드)`를 넣고, 가장 작은 거리 순으로 노드를 꺼냄.  
3. 꺼낸 노드의 인접 노드를 확인 → 더 짧은 경로 발견 시 테이블 갱신 + 큐에 삽입.  
4. 큐가 빌 때까지 2~3 반복.  

**시간 복잡도**  
- `O(E log V)` (간선 E, 노드 V)  

```
## 간단한 다익스트라 알고리즘 구현 

시간 복잡도: O($V^2$), V=노드의 개수 
특징: 각 단계마다 **'방문하지 않은 노드 중에서 최단 거리가 가장 짧은 노드를 선택'하기 위해 매 단계마다 1차원 리스트의 모든 원소를 확인 (순차 탐색)** 한다. 

다음 소스코드에서는 입력되는 데이터의 개수가 많다는 가정하에 파이썬 내장 함수인 input()을 더 빠르게 동작하는 sys.std.readlin()으로 치환하여 사용하는 방법을 적용했다. 또한 모든 리스트는 (노드의 개수 +1)의 크기로 할당하여, 노드의 번호를 인덱스로 하여 바로 리스트에 접근할 수 있도록 했다. 

```python
import sys 

input = sys.stdin.readline 
INF = int(1e9)

# n=노드 개수, m=간선 개수 
n, m = map(int, input().split())
start = int(input())


visited = [False] * (n+1)
# 최단 거리 테이블 무한으로 초기화 
distance = [INF] * (n+1)

# 그래프 입력 받기 
graph = [[] for i in range(n+1)]
for i in range(m):
    # a에서 b노드로 가는 비용이 c 
    a, b, c = map(int, input().split())
    graph[a].append((b, c))


def get_smallest_node():
    min_value = INF 
    index = 0 # 가장 최단 거리가 짧은 노드 (인덱스)
    for i in range(1, n+1):
        if distance[i] < min_value and not visited[i]:
            min_value = distance[i]
            index = i 
    return index 


def dijkstra(start):
    # 시작 노드에 대해서 초기화 
    distance[start]= 0
    visited[start] = True 

    for j in graph[start]:
        distance[j[0]] = j[1] 


    # 시작 노드를 제외한 n-1개의 노드에 대해 반복 
    for i in range(n-1):
        # 현재 최단 거리가 가장 짧은 노드를 꺼내서, 방문 처리 
        now = get_smallest_node()
        visited[now] = True 

        # 현재 노드와 다른 연결된 노드 확인 
        for j in graph[now]:
            cost = distance[now] + j[1]
            
            # 현재 노드를 거쳐서 다른 노드로 이동하는 거리가 더 짧은 경우 
            if cost < distance[j[0]]:
                distance[j[0]] = cost 


# 다이익스트라 알고리즘 수행 
dijkstra(start)

# 모든 노드로 가기 위한 최단 거리 출력 
for i in range(1, n+1):
    # 도달할 수 없는 경우 
    if distance[i] == INF:
        print('INFINITY')
    else:
        print(distance[i])
```

```md
예시 그래프) 
6 11
1
1 2 2
1 3 5
1 4 1
2 3 3
2 4 2
3 2 3
3 6 5
4 3 3
4 5 1
5 3 1
5 6 2

출력 예시: 
0
2
3
1
2
4
```

전체 노드 개수가 5000개 이하라면, 일반적으로 이 코드로 문제를 해결할 수 있지만, 노드의 개수가 10,000개를 넘어가면 이 코드로 문제를 해결하기 어렵다. 따라서 개선된 다익스트라 알고리즘을 이용해서 풀어야한다. 

## 개선된 다익스트라 알고리즘 구현 

개선된 다익스트라 알고리즘을 사용하면, 최악의 경우에도 시간 복잡도 O(ElogV)를 보장하여 해결할 수 있다. 간단한 다익스트라 알고리즘은 '최단 거리가 가장 짧은 노드'를 찾기 위해 매번 최단 거리 테이블을 선형적으로 탐색했다. 이 과정에서 O(V)의 시간이 걸리는데, 선형적인 방법이 아니라 더욱 빠르게 찾아 시간 복잡도를 줄인다. 즉, 힙(heap)이라는 자료구조를 사용하여 특정 노드까지의 최단 거리에 대한 정보를 빠르게 찾을 수 있다.heap을 사용하면 로그 시간이 걸린다. N=1,000,000일 때 $\log_{2} N$이 약 20인 것을 감안하면 속도가 획기적으로 빨라지는 것임을 이해할 수 있다. 

### 힙 (Heap) 자료 구조 

힙 자료구조는 우선순위 큐(Priority Queue)를 구현하기 위하여 사용하는 자료구조 중 하나이다. 큐 자료 구조는 가장 먼저 삽입한 데이터를 가장 먼저 삭제한다. 우선순위 큐는 **우선순위가 가장 높은 데이터를 가장 먼저 삭제한다**는 점이 특징이며 **우선순위 큐 구현에 최적화된 트리 기반 자료구조**이다. 우선순위 큐는 데이터를 우선순위에 따라 처리하고 싶을 때 사용한다. 예르 들어, 여러 개의 물건 데이터를 자료 구조에 넣었다가 가치가 높은 물건 데이터부터 꺼내서 확인해야하는 경우를 가정해보자. 

|자료구조| 추출되는 데이터|
|---|---|
|Stack| 가장 나중에 삽입된 데이터|
|Queue| 가장 먼저 삽입된 데이터|
|Priority Queue| 가장 우선순위가 높은 데이터|

파이썬에서는 우선순위 큐가 필요할 때 `PriorityQueue` 혹은 `heapq`를 사용할 수 있는데, 이 두 라이브러리는 모두 우선순위 큐 기능을 지원한다. 다만, PriorityQueue보다는 heapq가 더 빠르게 동작하기 때문에 수행 시간이 제한된 상황에서는 heapq를 사용하는 것을 권장한다. 

우선순위 값을 표현할 때는 일반적으로 정수형 자료형의 변수가 사용된다. 예를 들어 물건 정보가 있고, 이 물건 정보는 물건의 가치와 물건의 무게로만 구성된다고 가정해보자. 그러면 모든 물건 데이터를 (가치, 물건)으로 묶어서 우선순위 큐 자료구조에 넣을 수 있다. 이후에 우선순위 큐에서 물건을 꺼내게 되면, 항상 가치가 높은 물건이 먼저 나오게 된다. (우선순위 큐가 최대 힙 (max heap)으로 구현되어 있을 때 가정). 대부분의 프로그래밍 언어에서는 우선순위 큐 라이브러리의 데이터의 묶음을 넣으면, **첫 번째 원소**를 기준으로 우선순위를 설정한다. 따라서 데이터가 (가치, 물건)으로 구성된다면 '가치' 값이 우선순위 값이 되는 것이다. 이는 파이썬에서도 마찬가지이다. 

또한 우선순위 큐를 구현할 때는 내부적으로 최소 힙 (Min Heap) 혹은 최대 힙 (Max Heap)을 이용한다. 최소 힙을 이용하는 경우 '값이 낮은 데이터가 먼저 삭제'되며 최대 힙을 이용하는 경우 '값이 큰 데이터가 먼저 삭제'된다. 파이썬 라이브러리에서는 기본적으로 `Min Heap`을 이용하는데, 다익스트라 최단 경로 알고리즘에서는 비용이 적은 노드를 우선 방문하므로 최소 힙 구조를 기반으로 하는 파이썬의 우선순위 큐 라이브러리를 그대로 사용하면 적합하다. 

최소 힙을 최대 힙처럼 이용하려면 일부러 우순순위에 해당하는 값에 음수 부호(-)를 붙여서 넣었다가, 나중에 우선순위 큐에서 꺼낸 다음에 다시 음수 부호 (-)를 붙여서 원래의 값으로 돌리는 방식을 사용할 수 있다. 

|우선순위 큐 구현 방식| 삽입 시간| 삭제 시간|
|---|---|---|
|리스트| O(1) | O(N)|
|힙(Heap)| O(logN)| O(logN)|

데이터의 개수가 N개 일 때, 힙 자료구조
힙(Heap) 자료구조는, 
- 완전 이진 트리(Complete Binary Tree) 형태를 가지는 자료구조.
- 규칙: 부모 노드와 자식 노드 간에 우선순위 규칙이 있음. (불변식)
  - 최대 힙(Max Heap): 부모 ≥ 자식 (루트가 가장 큼).
  - 최소 힙(Min Heap): 부모 ≤ 자식 (루트가 가장 작음).
- 기본 연산:
  - 삽입 (push): 새 원소를 추가한 후 위로 올려 정렬해야함. 
  - 삭제 (pop): 루트 원소를 빼낸 후, 마자막 원소를 루트로 올려보내고 아래로 내려 정렬함. 

```{code-block} python
import heapq

# 빈 힙 생성
pq = []

# 삽입 (push)
heapq.heappush(pq, 5)
heapq.heappush(pq, 2)
heapq.heappush(pq, 8)

# 삭제 (pop, 가장 작은 값 반환)
print(heapq.heappop(pq))  # 2
```

#### 힙 문법 
```python

"""힙에 원소를 합입할 때는 heapq.heappush() 메서드를 사용하고, 힙에서 원소를 꺼내고자 할 때는 heapq.heappop() 메서드를 이용한다. 힙 정렬 (heap sort)을 heapq로 구현하는 예제를 통해 heapqd의 사용방법을 알아보자. 
"""
import heapq 

def heapsort(iterable):
  h = []
  result = []

  # 모든 원소를 차례대로 힙에 삽입 
  for value in iterable:
    heapq.heappush(h, value)
  # 힙에 삽입된 모든 원소를 차례대로 꺼내어 담기 
  for _ in range(len(h)):
    result.append(heapq.heappop(h))

  result = heapsort([1, 3, 5, 7, 9, 2, 4, 6, 8, 0])
  print(result)
```

#### 힙 구현

````{admonition} 부모, 자식 인덱스
:class: dropdown 

![17](../../assets/img/shortest_path/17.png)

- parent = (pos - 1) // 2 # (pos-1) >> 1 
- left = 2 * pos + 1
- right = 2*pos + 2 
````

````{admonition} min heap visualization 
:class: dropdown 

- `_siftdown`: 힙 연산에서 "내려보내기 (sift down)" 과정을 나타냄. 
    - min heap: 힙의 불변식 (invariant)은 "부모 노드가 자식 노드보다 작다"의 경우 
    - 새로운 원소를 힙에 삽입하면, 처음에는 마지막 (leaf) 자리에 들어가고, 그 원소가 부모보다 작은 경우에는 부모와 자리를 바꿈. 
        - 부모 > newitem -> 부모를 아래로 내린다. 
        - pos <- 부모 위치로 이동 
        - 즉, 부모들을 한 단계씩 밑으로 "내려보내고" 마지막에 newitem을 넣는 구조이다. 
    - 루트에 도달하거나 부모 <= newitem이면 멈춘다. 
    - 현재 위치에 newitem을 둔다. 
    - ex) [1, 3, 19, 2] -> 2 < 3 이므로  3을 아래로 내림. -> [1, 2, 19, 3]
- `_siftup`: 자식들을 따라 "위로 올려보내면서" 원소를 정리하는 과정 
    - 루트를 pop할때 마지막 원소를 루트 자리로 옮기고, 이제 루트에서부터 힙 규칙이 깨질 수 있으므로 아래로 내려가면서 정렬한다. 
    - root에서 시작해서 자식을 따라 내려가면서 위치를 바꾸고, 마지막에 newitem을 넣은 후 위로 올려보내는 과정 
    - 알고리즘  
        - root 자리에 new item을 둔다. 
        - 자식 노드 중 더 작은 자식을 고른다. 
        - 그 자식 < newitem이면, 자식을 위로 올린다.
            - pos <- 자식 위치로 이동 
        - 자식 >= newitem이거나 leaf에 도달하면 멈춘다. 
        - 현재 위치에 newitem을 둔다.
    - 예시 
        - step 1: pop root=1, last =3 -> 루트에 3 대입 [3, 2, 19]
        - step 3: 자식 중 작은 건 2. (2 < 3) 자식 2를 위로 올림. 

click the [link](https://www.cs.usfca.edu/~galles/visualization/Heap.html)

- _siftdown 예시 
    - 삽입: 5, 3, 8, 1, 6, 7, 2
- _siftup 예시
    - 현재 힙 [1, 2, 7, 3, 6, 8, 5] -> root 삭제 
    - buildHeap button -> remove smallest 

````
````{admonition} 힙 구현 
:class: dropdown 

```{code-block} python
---
caption: python standard library에서 제공하는 heapq의 구현 방식을 참고하였다. 
---
def heappush(heap, item):
    heap.append(item)
    _siftdown(heap, 0, len(heap) - 1)

def _siftdown(heap, root, pos):
    newitem = heap[pos] # newitem 값 저장 
    while pos > root:
        parent_pos = (pos-1) >> 1 
        if heap[parent_pos] > newitem: # 항상 newtiem과 부모를 비교해야함. 
            heap[pos] = heap[parent_pos] # 부모를 한 칸 내려보냄 
            pos = parent_pos
            continue 
        break 
    # SWAP 
    heap[pos] = newitem 

def heappop(heap):
    # 맨 "마지막 원소"를 제거: heap안의 원소가 하나라면 맨 처음 원소이고, 아니라면 root position에 갈 원소가 됨. 
    removed_item = heap.pop() # heap이 비어있으면 에러를 일으킴 
    if heap: # pop 이후에도 heap안에 item이 있는 경우 
        returnitem = heap[0]
        heap[0] = removed_item 
        _siftup(heap, 0)
        return returnitem 
    return removed_item 

def is_leaf(heap, pos):
    # time complexity for len(list) = O(1), 내부에 길이를 따로 저장하고 있음.
    # "왼쪽 자식 인덱스가 배열 길이보다 크거나 같은 경우"로 판정하면 충분 
    return (pos * 2) + 1 >= len(heap)
  

def _siftup(heap, pos):
    end = len(heap)
    newitem = heap[pos]
    child = 2* pos + 1 # 왼쪽 자식 

    while child < end: # is_leaf()와 동일한 형식   
        right= child + 1 
        # 더 작은 자식을 child로 선택 
        if right < end and heap[right] < heap[child]:
            child = right 
        
        # child가 newitem보다 작으면 child를 끌어올리고, pos를 child로 이동 
        if heap[child] < newitem:
            heap[pos] = heap[child] # child를 siftup 
            pos = child 
            child = 2 * pos + 1 
        else:
            break # newitem이 들어갈 자리이면 멈춤 

    
    heap[pos] = newitem 
    _siftdown(heap, 0, pos)

if __name__ == "__main__":
    heap = []
    heappush(heap, (1, 2))
    heappush(heap, (3, 1))
    heappush(heap, (19, 1))
    heappush(heap, (2, 1))

    for i in range(len(heap)):
        print(heappop(heap))

```
````


### 우선순위 큐를 이용한 단계별 문제 풀이 

**Step 1**

1번 노드가 출발 노드인 경우를 가정했을 때, 앞의 과정과 다른 것은 우선순위 큐를 따로 만들어 1번 노드를 넣는 것이다. 파이썬에서는 간단히 튜플 (0, 1)을 우선순위 큐에 넣는다. 파이썬의 heapq 라이브러리는 원소로 튜플을 입력받으면 **튜플의 첫 번째 원소를 우선순위 큐로 구성**한다. 따라서 (거리, 노드 번호) 순서대로 튜플 데이터를 구성해 우선순위 큐에 넣으면 거리순으로 정렬된다. 

다시 말하지만, 다익스트라는 **현재까지 발견된 최단 거리 후보들**을 계속 관리하는 알고리즘이다. 현재 방문하는 노드의 모든 인접 노드를 확인하고, **짧아진 경우에만** 전부 큐에 넣어야 합니다.

![8](../../assets/img/shortest_path/8.png)

**Step 2**
우선순위 큐를 이용하고 있으므로 거리가 가장 짧은 노드를 선택하기 위해서는 우선순위 큐에서 그냥 노드를 꺼내면 된다. 따라서 우선순위 큐에서 노드를 꺼낸 뒤에 **해당 노드를 이미 처리한 적이 있다면** 무시하면 되고, 아직 처리하지 않은 노드에 대해서만 처리하면 된다. Step 1의 우선순위 큐에서 원소를 꺼내면 (0, 1)이 나오고, 1번 노드를 거쳐서 2번, 3번, 4번 노드로 가는 최소 비용을 계산한다. 각각 테이블 값을 갱신한 후, 더 짧은 경로를 찾은 노드 정보들은 다시 우선순위 큐에 넣는다. 
![9](../../assets/img/shortest_path/9.png)

**Step 3**
다음으로 (1, 4)의 값을 갖는 원소가 우선순위 큐에서 추출되며, 아직 노드 4를 방문하지 않았고 현재 최단 거리가 가장 짧은 노드가 4이다. 따라서 노드 4를 기준으로 연결된 간선들을 확인한다. 4번 노드를 거쳐서 3번과 5번 노드로 가는 최소 비용은 차례대로 4와 2이다. 이는 기존의 리스트에 다겨있던 값들보다 작기 때문에, 리스트의 값을 갱신하고 우선순위 큐에 두 원소 (4, 3), (2, 5)를 추가로 넣어준다. 
![10](../../assets/img/shortest_path/10.png)

**Step 4**

우선순위 큐에서 원소를 pop하면 노드 2가 꺼내진다. 2번 노드를 거쳐서 가는 경우 중 다음 노드에서 현재의 최단 거리를 더 짧게 갱신할 수 있는 방법은 없다. 따라서 우선순위 큐에 어떠한 원소도 들어가지 않고 다음과 같이 리스트가 갱신된다. 

![11](../../assets/img/shortest_path/11.png)

**Step 5**

이번 단계에서는 노드 5에 대해 처리한다. 5번 노드를 거쳐서 3번과 6번 노드로 갈 수 있다. 현재 5번 노드까지 가는 최단 거리가 2이므로 5번 노드에서 3번 노드로 가는 거리인 1을 더한 3이 기존의 값인 4보다 작다. 따라서 새로운 값인 3으로 갱신한다. 또한 6번 노드로 가는 최단 거리 역시 마찬가지로 갱신된다. 그래서 이번에는 (3, 3)과 (4, 6)이 우선순위 큐에 들어간다. 우선순위 큐를 보면, 작을 때마다 원소를 넣어주는데, 같은 노드이지만 이전에 넣어서 (거리) 값이 더 큰 원소가 있음을 확인할 수 있다. 따라서, heappop()으로 원소를 빼준 후, 가장 최신으로 업데이트된 값보다 큰 경우에는 무시하고 넘어가주면 된다. 

![12](../../assets/img/shortest_path/12.png)

**Step 6**

우선순위 큐에서 노드 3을 빼고, 3에서는 노드 2번과 6번으로 갈 수 있다. 최단 거리 테이블은 갱신되지 않으며 따라서 우선순위 큐에도 아무것도 넣어주지 않는다. 

![13](../../assets/img/shortest_path/13.png)

**Step 7**

원소 (4, 3)을 꺼낸다. 다만, 3번 노드는 현재 최단 거리 테이블의 값보다 크므로 이 원소는 무시한다. 

![14](../../assets/img/shortest_path/14.png)

**Step 8**

이어서 원소 (4, 6)이 꺼내진다.

![15](../../assets/img/shortest_path/15.png)


**Step 9**

마지막으로 남은 원소를 꺼내지만, 아까와 마찬가지로 이미 처리된 노드이므로 무시한다. 

![16](../../assets/img/shortest_path/16.png)

### 개선된 dijkstra 구현 

````{admonition} heapq를 이용한 dijkstra 구현 
:class: dropdown 

```{code-block} python
import heapq 
import sys 
input = sys.stdin.readline 
INF = int(1e9)

# 노드의 개수, 간선의 개수를 입력받기 
n, m = map(int, input().split())
# 시작 노드 번호 입력 받기 
start = int(input())
 
# 각 노드에 연결되어 있는 노드에 대한 정보를 담는 리스트 만들기 
graph = [[] for i in range(n+1)]
# 최단 거리 테이블을 모두 무한으로 초기화 
distance = [INF] * (n+1)

# 모든 간선 정보 입력받기 
for _ in range(m):
    a, b, c= map(int, input().split())
    # a번 노드에서 b번 노드로 가는 비용이 c 
    graph[a].append((b, c))

def dijkstra(start):
    q = [] 

    distance[start] = 0
    heapq.heappush(q, (0, start))

    while q:
        cur_dis, node = heapq.heappop(q)

        if cur_dis > distance[node]:
            continue 

        for weight, nxt_node in graph[node]:
            if cur_dis + weight < distance[nxt_node]:
                distance[nxt_node] = cur_dis + weight # 현재 node까지 온 비용 + nxt_node로 가는 비용 
                heapq.heappush(q, (distance[nxt_node], nxt_node))

# 다익스트라 알고리즘 수행 
dijkstra(start)

# 모든 노드로 가기 위한 최단 거리 출력 
for i in range(1, n+1):
    if distance[i] == INF:
        print("INF")
    else:
        print(distance[i])

```
````

## Floyd-Warshall Algorithm

플로이드 워셜 알고리즘은 "모든 지점에서 다른 모든 지점까지의 최단 경로를 모두 구해야 하는 경우"에 사용할 수 있는 알고리즘이다. 다익스트라 알고리즘은 단계마다 최단 거리를 가지는 노드를 하나씩 반복적으로 선택한다. 그리고 해당 노드를 거쳐 가는 경로를 확인하며, 최단 거리 테이블을 갱신하는 방식으로 동작한다. 플로이드 워셜 알고리즘 또한 단계마다 '거쳐 가는 노드'를 기준으로 알고리즘을 수행한다. 하지만 **매번 방문하지 않은 노드 중에서 최단 거리를 갖는 노드를 찾을 필요가 없다**는 점이 다르다. 

노드의 개수가 N개 일 때 알고리즘상 N번의 단계를 수행하며, 단계마다 O($N^2$)의 연산을 통해 '현재 노드를 거쳐 가는' 모든 경로를 고려한다. 따라서 총 **시간 복잡도는 O($N^3$)**이다. 

플로이드 워셜 알고리즘은 **2차원 리스트**에 '최단 거리'정보를 저장한다. 다익스트라 알고리즘은 그리디 알고리즘인 반면, 플로이드 워셜 알고리즘은 **다이나믹 프로그래밍**이다. 노드의 개수가 N개 일 때, N번 만큼의 단계를 반복하며 '점화식에 맞게' 2차원 리스트를 갱신하기 때문이다. 

## 플로이드 워셜 알고리즘 핵심 아이디어 



