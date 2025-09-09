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

# Lecture 1-2. DFS/BFS 

그래프를 탐색하기 위한 대표적인 두 가지 알고리즘인 `DFS (Depth First Search)`와 `BFS (Breadth First Search)`에 대해서 학습한다. `탐색 (Search)`이란 많은 양의 데이터 중에서 원하는 데이터를 찾는 과정을 말한다. 이 개념을 위해서 사전적으로 `자료 구조 (Data structure)` `Stack`, `Queue`, `Recursive function`를 간단히 정리하고자 한다. 

## Preliminaries 
스택과 큐를 이해하기 위해서 두 핵심적인 함수 (push, pop) 및 underflow, overflow 개념을 알아야한다. 

- 삽입 (Push): 데이터를 삽입한다. 
- 삭제 (Pop): 데이터를 삭제한다. 
- 오버플로 (Overflow): 자료구조가 수용할 수 있는 데이터의 크기를 이미 가득 찬 상태에서 삽입 연산을 수행할 때 발생. 
- 언더플로 (Underflow): 데이터가 전혀 없는 상태에서 삭제 연산 수행할 때 발생 
  
### Stack 

스택은 `선입후출 (First In Last Out)` 또는 `후입선출 (Last In Last Out)`라고 한다. 파이썬에서는 스택을 이용할 때 별도의 라이브러리가 아닌 기본 리스트에서 `append()`와 `pop()` 메서드를 사용하면 된다. 

```{grid} 2
:gutter: 2

:::{grid-item}
:columns: 6
![DFS 예시](../assets/img/DFS_BFS/1.png)
:::

:::{grid-item}
:columns: 6
![BFS 예시](../assets/img/DFS_BFS/2.png)
:::
```

```{code-block} python
---
caption: "stack 구현 예시"
----
stack = [] 
# 삽입 (5) - 삽입 (2) - 삽입 (3) - 삽입 (7) - 삭제 () - 삽입 (1) - 삽입 (4) - 삭제 ()

stack.append(5)
stack.append(2)
stack.append(3)
stack.append(7)
stack.pop()
stack.append(1)
stack.append(4)
stack.pop()

print(stack) # 최하단 원소부터 출력, [5, 2, 3, 1]
print(stack[::-1]) # 최상단 원소부터 출력 [1, 3, 2, 5]
```
### Queue 

큐 (Queue)는 대기줄에 비유할 수 있다. 먼저 온 사람이 먼저 들어가게된다. 이러한 구조를 `선입선출 (First In First Out, FIFO)`구조라고 한다. 파이썬으로 queue를 구현할 때는 collections 모듈에서 제공하는 deque 자료구조를 사용하면 된다. 또한 deque 객체를 리스트 자료형으로 변경하고자 하면, list() 메서드를 이용하면 된다. (list(queue))

```{grid} 2
:gutter: 2

:::{grid-item}
:columns: 6
![DFS 예시](../assets/img/DFS_BFS/3.png)
:::

:::{grid-item}
:columns: 6
![BFS 예시](../assets/img/DFS_BFS/4.png)
:::
```

```{code-block} python
---
caption: "queue 구현 예시" 
---
from collections import deque 

# Queue 구현을 위해 deque 라이브러리 사용 
queue = deque()

#삽입(5) - 삽입(2) - 삽입(3) - 삽입(7) - 삭제() - 삽입(1) - 삽입(4) - 삭제()
queue.append(5)
queue.append(2)
queue.append(3)
queue.append(7)
queue.popleft()
queue.append(1)
queue.append(4)
queue.popleft()

print(queue) # 먼저 들어온 순서대로 출력 (3,7,1,4)
queue.reverse() # 다음 출력을 위해 역순으로 바꾸기 
print(queue) # 나중에 들어온 원소부터 출력 (4, 1, 7, 3)
```

### Recursive Function 

`재귀 함수 (Recursive Function)`란 자기 자신을 다시 호출하는 함수를 의미한다. 다음은 재귀 함수의 예시를 살펴보자. 

```{code-block} python
---
caption: "종료 조건 없는 재귀 함수 간단 예시"
---
def recursive_function():
    print('재귀 함수를 호출합니다')

recursive_function() 
```

이 코드를 실행하면 '재귀 함수를 호출합니다'라는 문자열을 무한히 출력한다. 어느 정도 출력 후 다음과 같은 오류 메시지가 출력된다. 

```{code-block} text
---
caption: "오류 메시지"
---
RecursionError: maximum recursion depth exceeded while pickling an object
```

이 오류 메시지는 재귀(recursion)의 최대 깊이를 초과했다는 내용으로 함수 `종료 조건`을 항상 명시해야한다. 

#### 재귀 함수의 종료 조건 
무한 호출을 막기위해 재귀 함수에서는 종료 조건을 항상 명시해야한다. 

```{code-block} python 
---
caption: "재귀 함수 종료 예제"
---

def recursive_function(i):
    # 100번째 출력했을 때 종료되도록 종료 조건 명시 
    if i == 100:
        return 
    
    print(i, '번째 재귀 함수에서', i+1, '번째 재귀 함수를 호출합니다.')
    recursive_function(i+1)
    print(i, '번째 재귀 함수를 종료합니다.')

recursive_function(1)
```

```{code-block} text
---
caption: "출력 예시"
---
1 번째 재귀 함수에서 2 번째 재귀 함수를 호출합니다.
2 번째 재귀 함수에서 3 번째 재귀 함수를 호출합니다.
...
99 번째 재귀 함수에서 100 번째 재귀 함수를 호출합니다.
99 번째 재귀 함수를 종료합니다.
98 번째 재귀 함수를 종료합니다.
...
2 번째 재귀 함수를 종료합니다.
1 번째 재귀 함수를 종료합니다.
```

컴퓨터 내부에서 `재귀 함수의 수행`은 `스택 자료구조 (Stack)`를 이용한다. 함수를 계속 호출했을 때 가장 마지막에 호출한 함수가 먼저 수행을 끝내야 그 앞의 함수 호출이 종료되기 때문이다. 즉, 재귀 함수는 내부적으로 스택 자료구조와 동일하다는 것만 기억하자. 따라서 스택 자료 구조를 활용해야하는 상당수 알고리즘은 재귀 함수를 이용해서 간편하게 구현될 수 있다. DFS가 가장 대표적인 알고리즘이다. 

`팩토리얼 (Factorial)` 문제를 재귀 함수를 이용해서 풀 수 있다. 

```{code-block} python 
---
caption: "2가지 방식으로 구현한 팩토리얼 예제"
---
# 반복적으로 구현한 n! 
def factorial_iterative(n):
    result = 1 
    # 1부터 n까지의 수를 차례대로 곱하기 
    for i in range(1, n+1):
        result *= i 
    return result 

# 재귀적으로 구현한 n! 
def factorial_recursive(n):
    if n <= 1:
        return 1 
    
    # n! = n * (n-1)! 
    return n * factorial_recursive(n-1)

print('반복적으로 구현:', factorial_iterative(5)) # 반복적으로 구현: 120
print('재귀적으로 구현:', factorial_recursive(5)) # 재귀적으로 구현: 120
```

재귀적으로 구현하면 코드가 더 간결해진다. 재귀 함수는 수학의 `점화식 (recurrence relation, recursive formula)`을 그대로 소스코드로 옮겼기 때문에 더 간결하다. 수학에서 점화식이란 특정한 함수를 자신보다 더 작은 변수에 대한 함수와의 관계로 표현한 것을 의미한다. 

보통 점화식에서 `종료 조건(base case)`을 찾을 수 있는데, 팩토리얼 문제에서 종료 조건은 'n이 0 혹은 1'경우 이다. 따라서 재귀 함수 내에서 특정 조건일 때 `더 이상 재귀적으로 호출하지 않고 종료(terminate without further recursive calls)`하도록 if 문을 이용하여 꼭 종료 조건을 구현해주어야 한다. 

```{admonition} 정리
:class: important 
- Stack는 First In Last Out, Queue는 Last In Last Out 
- 재귀 함수 (Recursive function)은 스택으로 생각할 수 있는데, 가장 마지막에 호출한 함수가 먼저 수행을 끝내야 그 앞의 함수 호출이 종료된다. 
- 따라서, 종료 조건 (base case)에서 반드시 재귀 함수를 호출하지 않고 종료 조건을 만들어줘야한다. 
```

### Graph 

그래프는 `노드 (Node)`와 `간선 (Edge)`로 표현되며 이때 노드를 `정점(Vertex)`이라고도 말한다. `그래프 탐색 (Graph search)`은 하나의 노드를 시작으로 다수 의 노드를 방문하는 것을 말한다. 두 노드가 간선으로 연결되어 있으면 '두 노드는 `인접하다(Adjacent)`'라고 표현한다. 

```{grid} 2
:gutter: 2

:::{grid-item}
:columns: 6
![그래프 예시](../assets/img/DFS_BFS/5.png)
::: 

:::{grid-item}
:columns: 6
![Adjacent Matrix & Adjacent List](../assets/img/DFS_BFS/6.png)
:::
```

그래프는 크게 2가지 방식으로 표현할 수 있다. 

1. 인접 행렬 (Adjacent Matrix): 2차월 배열로 그래프의 연결 관계 표현 
2. 인접 리스트 (Adjacent List): 리스트로 그래프의 연결 관계를 표현 

우선 `인접 행렬 (Adjacent Matrix)` 방식은 파이썬에서는 `2차원 리스트`로 구현할 수 있다. 연결이 되어 있지 않은 노드끼리는 `무한 (Infinity)` 의 비용이라고 작성한다. 실제 코드에서는 논리적으로 정답이 될 수 없는 큰 값 중에서 999999999 등의 값으로 `초기화 (Initialization)`하는 경우가 많다. 

```{code-block} python
---
caption: 인접 행렬 방식 예제 
---

INF = 9999999999 # 무한의 비용 선언 

graph = [
    [0, 7, 5],
    [7, 0, INF],
    [5, INF, 0]
]
print(graph)
```
```{code-block} text
[[0, 7, 5], [7, 0, 9999999999], [5, 9999999999, 0]]
```

`인접 리스트 (Adjacent List)` 방식은 모든 노드에 연결된 노드에 대한 정보를 차례대로 연결하여 저장한다. 인접 리스트는 `연결 리스트 (Linked List)`라는 자료구조를 이용해 구현하고, C++/Java 와 같은 언어에서는 연결 리스트 기능을 위한 표준 라이브러리를 제공한다. 파이썬에서는 리스트 자료형을 사용해 단순히 2차원 리스트를 이용하면 된다. 

```{code-block} python
---
caption: "인접 리스트 방식 예제"
---
# 행(Row)이 3개인 2차원 리스트로 인접 리스트 표현 
graph = [[] for _ in range(3)]

# 노드 0에 연결된 노드 정보 저장 (노드, 거리)
graph[0].append((1, 6))
graph[0].append((2, 5))

# 노드 1에 연결된 노드 정보 저장 (노드 ,거리)
graph[1].append((0, 7))

# 노드 2에 연결된 노드 정보 저장 (노드 ,거리)
graph[2].append((0, 5))

print(graph)
```
```{code-block} text 
[[(1, 6), (2, 5)], [(0, 7)], [(0, 5)]]
```





## 탐색 알고리즘 DFS/BFS 

### DFS (Depth-First Search)

`DFS`는 깊이 우선 탐색이라고도 부르며, 그래프에서 깊은 부분을 우선적으로 탐색하는 알고리즘이다. 

### BFS 

## 예제 

### 음료수 얼려 먹기 

### 감시 피하기 

### 블록 이동하기 