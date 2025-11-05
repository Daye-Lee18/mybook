# Lecture 5-2. Shortest Path & Heap 

````{admonition} Heap vs. sort 
:class: dropdown 

Shortest path를 dijkstra로 찾는 과정에서 우리는 heapq 라이브러리를 사용하여 priority queue구조에 대하여 배웠었다. priority queue는 sort()와 다르게, '최대/최소'값을 반복적으로 찾아야하는 문제에서 가장 효과적이다. 특히, 데이터가 실시간으로 나갔다 들어오는 경우 Sort()보다 더 유용하다. 

sort()와 min-heap은 오름차순으로 두면, "매번 현재 가장 싼 간선"을 선택하므로 정확성은 동일하다. 또한 시간 복잡도는 
- 정렬: O(E logE)
- min-heap: heapify O(E) + E번 pop (logE씩) -> O(ElogE)
위와 같이 이론상 동일하지만, 실무에서는 정렬이 상수항이 더 작고 캐시 친화적이라 빠른 일이 많다. (Timsort). 반면 min-heap은 매 pop마다 log 비용이 들어서 오히려 느릴 수 있다. 

또한 메모리 측면에서 둘 다 간선 E개 저장이 필요하므로 O(E)로 동일. 

보통의 경우 sort()를 사용하고, min-heap은 다음과 같은 특수 상황에서 사용하면 효율적이다. 
- 간선이 스트리밍으로 들어오거나 한번에 다 만들기 어려운 상황 (외부 입력/온라인 처리)
- 가장 싼 간선부터 일부만 처리하며 조기 종료가 확실한 특수 케이스 
````

예시 문제 링크 
- [가로등 설치](https://www.codetree.ai/ko/frequent-problems/samsung-sw/problems/street-light-installation/description)
- [코드 트리 채점기](https://www.codetree.ai/ko/frequent-problems/samsung-sw/problems/codetree-judger/description)
- [코드트리 투어](https://www.codetree.ai/ko/frequent-problems/samsung-sw/problems/codetree-tour/description)
- [해적 선장 코디](https://www.codetree.ai/ko/frequent-problems/samsung-sw/problems/pirate-captain-coddy/description)
- [개구리의 여행](https://www.codetree.ai/ko/frequent-problems/samsung-sw/problems/frog-journey/description)

## 가로등 설치 

`````{admonition} Lazy Deletion
:class: dropdown 

***지연 갱신(Lazy Deletion)*** 이란, 힙에서 특정 원소를 찾아 삭제할때 바로 삭제하지 않고 오래된 정보를 그냥 두되, 힙에서 원소를 꺼낼 때마다 "이 정보가 현재 유효한가?"를 검사하는 것이다. 즉, 해당 문제에서 가로등이 제거되면, 그와 관련된 도로나 위치 정보가 힙에 여전히 남아있게 되는데, 힙에서 특정 원소를 찾아 삭제하는 것이 비효율적이므로 힙에서 꺼낼 때 정보가 유효한지 확인하는 것이다. 유효하지 않다면 버리고 다음 원소를 꺼내는 방식으로 처리하면, 전체적인 효율성을 유지하면서 로직을 간단하게 만들 수 있다. 

dijkstra algorithm의 개선된 코드 버전에서 현재 꺼낸 노드로 가는 비용이 최소가 아니면 skip하는 로직과 동일하다. 

해당 문제에서는 `lamp_pos` 리스트에 각 가로등의 ID를 인덱스로 하여 위치 (pos)를 저장한다. 따라서 O(1)시간에 특정 가로등의 위치를 조회할 수 있다. 가로등이 제거되면 해당 위치를 -1와 같은 무효한 값으로 표시하여 '지연 갱신'에 사용된다. 

예를 들어, 두 가로등 사이에 다른 가로등이 추가되거나 제거되었다면, 해당 정보를 "실시간으로" 우선 lamp_pos에서 추가하거나 제거한다. 그러나 heapq에 있는 정보는 그대로 두고 나중에 실제로 있는지는 계속 실시간 업데이트가 되는 `lamp_pos`에서 확인하는 것이다. 

```{code-block} python
import heapq 

roads = []
lamp_pos = []

def get_max_road():
    while roads:
        road = roads[0]
        # 힙에서 꺼낸 도로 정보가 현재 가로등 위치와 일치하는지 확인
        # 즉, 이 도로를 구성하는 두 가로등이 여전히 인접해 있는지 검사
        if lamp_pos[road.left_lamp_id] == road.st_pos and lamp_pos[road.right_lamp_id] == road.st_pos + road.length:
            break
        heapq.heappop(roads)
```

다음 [프로그래머스 문제 링크](https://school.programmers.co.kr/learn/courses/30/lessons/42628) / (LeetCode 295)[https://leetcode.com/problems/find-median-from-data-stream/description/] / Baekjoon 7662 이중 우선순위큐에서 lazy deletion을 연습할 수 있다. 

````{admonition} 인덱스를 통한 풀이 
:class: dropdown 

삽입 순서를 ID로 지정한 후, 실시간 deleted 리스트 인덱스를 ID로 하여 lazy deletion 표시 배열 사용. 

```{code-block} python 
import heapq

def solution(operations):
    queue_b = []
    queue_s = []
    deleted = [0]*len(operations)

    for i, op in enumerate(operations): 
        cm, d = op.split()
        num = int(d)

        if cm == "I": 
            heapq.heappush(queue_b, (-num, -i))  # max heap
            heapq.heappush(queue_s, (num, i))   # min heap

        elif cm == "D" and num == 1:
            while queue_b:
                deleted_n, j = heapq.heappop(queue_b)
                if deleted[abs(j)] != 1: 
                    deleted[abs(j)] = 1
                    break

        elif cm == "D" and num == -1:
            while queue_s:
                deleted_n, j = heapq.heappop(queue_s)
                if deleted[j] != 1: 
                    deleted[j] = 1
                    break

    max_v = None
    while queue_b:
        val, j = heapq.heappop(queue_b)
        if deleted[abs(j)] == 0:
            max_v = -val
            break

    min_v = None
    while queue_s:
        val, j = heapq.heappop(queue_s)
        if deleted[j] == 0:
            min_v = val
            break

    if max_v is None or min_v is None:
        return [0, 0]
    else:
        return [max_v, min_v]
```
````

````{admonition} 객체 공유를 이용한 풀이 
:class: dropdown 

````{code-block} python 
import heapq 

class Num:
    def __init__(self, num, valid=True):
        self.num = num
        self.valid = valid

    def __lt__(self, other):
        return self.num < other.num # min_heap 
    
def check_valid(heap):
    while len(heap) != 0:
        cur_num = heap[0]
        if not cur_num.valid:  # 없어진 값이면 지우고 다시 pop 
            heapq.heappop(heap)
        else:
            cur_num.valid = False
            heapq.heappop(heap)
            break 

def check_max_valid(heap):
    while len(heap) != 0:
        neg_num, num_obj = heap[0]
        if not num_obj.valid:  # 이미 없어진 값이면 지우고 다시 pop 
            heapq.heappop(heap)
        else:
            num_obj.valid = False
            heapq.heappop(heap)
            break 

def clean_min(heap):
    while heap and not heap[0].valid:
        heapq.heappop(heap)

def clean_max(max_heap):
    while max_heap and not max_heap[0][1].valid:
        heapq.heappop(max_heap)

def solution(operations):
    min_heap = [] 
    max_heap = []
    for op in operations:
        char, num = op.split()
        if char == 'I':
            same_instance = Num(int(num))
            heapq.heappush(min_heap, same_instance)
            heapq.heappush(max_heap, (-same_instance.num, same_instance))
        elif char == 'D' and num == '1': # 최댓값 삭제 
            # 예를 들어, min_heap에서 먼저 지워진 상황이지만, max_heap에서 아직 안지워진 경우 
            check_max_valid(max_heap)
        else:
            # 최솟값 삭제 
            # 예를 들어, max_heap에서 먼저 지워진 상황이지만, min_heap에서 아직 안지워진 경우 
            check_valid(min_heap)

    # 실제 비어있는지 정리용 
    clean_min(min_heap)
    clean_max(max_heap)

    return [0,0] if len(min_heap) == 0 else [max_heap[0][0]* -1, min_heap[0].num]


# operations = ["I 16", "I -5643", "D -1", "D 1", "D 1", "I 123", "D -1"]
# operations = ["I -45", "I 653", "D 1", "I -642", "I 45", "I 97", "D 1", "D -1", "I 333"]
# print(solution(operations))

````

위의 두 방식 중에서 대부분 온라인 저지에서는 시간이 보틀넥이기 때문에, index를 활용한 deleted list를 이용하는 방식이 베스트 선택이다. 
`````

````{admonition} class comparison for heap
:class: dropdown 

현재 가로등 정보에서 관련 정보, 왼쪽 가로등, 오른쪽 가로등과 같은 가로등 쌍에 대한 정보를 담는 class를 heap에서 정렬하기 위해 __lt__ 비교 연산자 정의가 필요하다. 이는 다음처럼 정의할 수 있다. 

```{code-block} python
import heapq 

roads = []  # 인접한 가로등 사이의 도로 정보를 저장할 힙 

# 인접한 가로등 사이의 도로 정보를 담는 클래스 
class Road:
    def __init__(self, left_lamp_id, right_lamp_id, length, st_pos):
        self.left_lamp_id = left_lamp_id #도로의 왼쪽 가로등 ID 
        self.right_lamp_id = right_lamp_id
        self.length = length 
        self.st_pos = st_pos # 도로 시작 위치 (왼쪽 가로등 ID)
    
    # 최대 힙으로 사용하기 위한 비교 연산자 정의 
    def __lt__(self, other):
        if self.length != other.length:
            return self.length > other.length 
        return self.st_pos < other.st_pos 

heapq.heappush(roads, Road(i-1, i, length, lamp_pos[i-1]))
```
````

`````{admonition} doubly linked list 
:class: dropdown 

보통 doubly linked list 는 pointer를 사용하여 연결하지만 "삽입/삭제" 기능에 있어서 최대 O(N) time complexity가 소요되는 단점이 있다. 이와 달리 prev_lamp_id와 next_lamp_id의 두 리스트를 만들어서 각 가로등의 ID를 인덱스로 하여 이전 가로등과 다음 가로등의 ID를 저장한다. 이는 가로등들을 위치 순서에 따라 이중 연결 리스트처럼 관리하기 위함이며, 가로등 제거 시 양옆의 가로등을 O(1)에 찾는데 사용된다. 

```python
prev = [-1, 0, 1, 2]
next = [1, 2, 3, -1]
```

위의 코드에서 i번째 노드 삭제는:
```python
prev[next[i]] = prev[i]
next[prev[i]] = next[i]
```
삭제 연산이 O(1)으로 줄어든다. 즉, linked list를 "배열로 구현하면" 사실상 매우 빠른 연결 리스트가 된다. 

````{admonition} DLL implementation with array 
:class: dropdown 

```{code-block} python 
MAX = int(1e9)
prev = [-1] * MAX 
next = [-1] * MAX 
value = [None] * MAX 

# head = some index 
# insert x after cur 
def insert(cur, x):
    next[x] = next[cur]
    prev[x] = cur 

    # NOTE: next node의 prev에 접근할때 
    if next[cur] != -1: # linked list의 last가 아닌 경우, 
        prev[next[cur]] = x 

    next[cur] = x 

def delete(x):
    if prev[x] != -1:
        next[prev[x]] = next[x] 
    if next[x] != -1:
        prev[next[x]] = prev[x]
```
````


해당 방식을 사용하여 빠르게 풀 수 있는 문제들을 아래에 적어놓았다. 

- [LRU Cache 최적 구현]
- [BOJ 5397 키로거]
- [BOJ 1406 에디터]
`````