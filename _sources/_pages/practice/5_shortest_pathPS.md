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


````{admonition} 전역변수 
:class: dropdown 

어떤 전역 변수를 사용할 것인지, 정리한후, 전역 변수를 한번에 INIT 해놓는다. 
```{code-block} python
# 전역 변수 선언
N, M = None, None
lamp_pos = None       # 각 가로등의 위치 (ID -> 위치)
next_lamp_id = None   # 각 가로등의 다음 가로등 ID (이중 연결 리스트 구현)
prev_lamp_id = None   # 각 가로등의 이전 가로등 ID (이중 연결 리스트 구현)
roads = []            # 도로 정보를 저장할 최대 힙
lamp_pos_min_heap = [] # 가로등 위치를 저장할 최소 힙 (가장 왼쪽 가로등 탐색용)
lamp_pos_max_heap = [] # 가로등 위치를 저장할 최대 힙 (가장 오른쪽 가로등 탐색용)
```
참고로, `if __name__ == "__main__"`은 함수 내부가 아니라 전역 스코프안에 있는 조건문이라 전역 스코프 (global scope)에 해당한다. 따라서, 이 안에서 `global`을 사용하면 문법 오류가 난다. 
````

`````{admonition} Lazy Deletion
:class: dropdown 

***지연 갱신(Lazy Deletion)*** 이란, 힙에서 특정 원소를 찾아 삭제할때 바로 삭제하지 않고 오래된 정보를 그냥 두되, 힙에서 원소를 꺼낼 때마다 "이 정보가 현재 유효한가?"를 검사하는 것이다. 즉, 해당 문제에서 가로등이 제거되면, 그와 관련된 도로나 위치 정보가 힙에 여전히 남아있게 되는데, 힙에서 특정 원소를 찾아 삭제하는 것이 비효율적이므로 힙에서 꺼낼 때 정보가 유효한지 확인하는 것이다. 유효하지 않다면 버리고 다음 원소를 꺼내는 방식으로 처리하면, 전체적인 효율성을 유지하면서 로직을 간단하게 만들 수 있다. 

dijkstra algorithm의 개선된 코드 버전에서 현재 꺼낸 노드로 가는 비용이 최소가 아니면 skip하는 로직과 동일하다. 

해당 문제에서는 `lamp_pos` 리스트에 각 가로등의 ID를 인덱스로 하여 위치 (pos)를 저장한다. 따라서 O(1)시간에 특정 가로등의 위치를 조회할 수 있다. 가로등이 제거되면 해당 위치를 -1와 같은 무효한 값으로 표시하여 '지연 갱신'에 사용된다. 

예를 들어, 두 가로등 사이에 다른 가로등이 추가되거나 제거되었다면, 해당 정보를 "실시간으로" 우선 lamp_pos에서 추가하거나 제거한다. 그러나 heapq에 있는 정보는 그대로 두고 나중에 실제로 있는지는 계속 실시간 업데이트가 되는 `lamp_pos`에서 확인하는 것이다. 

즉, 실시간으로 업데이트 되는 정보 `lamp_pos`, `prev`, `next`이고 지연되는 것은 priority queue 즉 `road_q`, `left_q`, `right_q`와 같은 정보이다. 

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

````{admonition} 좌/우 전역값 동기화 
:class: dropdown 

이론상 전역 변수로 끝값을 정리하면서 “삭제 때만 갱신”하는 것으로 코드를 구현할 수 있다. 이런 방식은 힙 사용때보다 메모리 사용량을 줄일 수 있지만, 업데이트 순서가 미묘하게 어긋나거나 버그가 하나만 생겨도 전역 끝값이 틀어질 수 있다. 전역 캐시는 모든 변화 경로를 100% 누락 없이 갱신해야 하는 부채가 생깁니다(연속 삭제, 경계 근처 혼합 연산, 예외 입력 등). 하지만 ***힙+지연검증*** 은 질의 시점에 항상 자기 교정을 하므로, 누락이 끼어도 pop하면서 복구돼요.
→ “가능은 하지만 유지비가 큰 전역 캐시” vs “조금 더 구조화된 힙 기반”의 선택 문제입니다.
````

````{admonition} ceiling 
:class: dropdown 

중간 좌표 계산에서 천장(ceil)인 경우에, 아래 코드를 사용한다. 

```python
new_pos = root.st_pos + (road.length + 1 ) // 2
```

2로 나누어 round()를 적용하면 파이썬에서는 '은행가 반올림(0.5를 짝수로)이라 4.5 -> 4, 5.5->6 같은 예외가 생긴다. 문제에서는 ceiling 을 요구하므로 위의 코드를 사용해야한다. 
````

````{admonition} solution
:class: dropdown 

```{code-block} python
import sys 
import heapq 

input = sys.stdin.readline
# sys.stdin = open('Input.txt')

'''
INIT:

1) road_q = 
2) left_q = 
3) right_q = 
4) prev = 
5) next = 
6) N = length of the road 
7) lamps_pos = 
'''

class Road:
    def __init__(self, length, start_pos, end_pos, left_num, right_num):
        self.length = length
        self.start_pos = start_pos 
        self.end_pos = end_pos 
        self.left_num = left_num 
        self.right_num = right_num

    def __lt__(self, other):
        if self.length == other.length:
            return self.start_pos < other.start_pos
        return self.length > other.length 

def check(light_nums, pos):
    poses = zip(pos[:-1], pos[1:])
    for idx, (left_pos, right_pos) in enumerate(poses):
        length = right_pos - left_pos 
        left_num = idx + 1 
        right_num = left_num + 1
        # Road 안에 __lt__로 length에 대해서 max_heap, left_pos에 대해서 min_heap으로 heapq에서 "정렬"되도록 해놓음. 
        heapq.heappush(road_q, Road(length, left_pos, right_pos, left_num, right_num))
        heapq.heappush(left_q, (left_pos, left_num))
        heapq.heappush(right_q, (-left_pos, left_num))
        lamps_pos.append(left_pos)
        prev.append(left_num-1 if left_num != 1 else -1)
        next.append(right_num) 
    
    # 맨 마지막 노드 
    heapq.heappush(left_q, (right_pos, right_num))
    heapq.heappush(right_q, (-right_pos, right_num))
    prev.append(right_num-1)
    next.append(-1)
    lamps_pos.append(right_pos)

 
def valid_check(cur_road):
    '''
    현재 두 가로등 사이의 정보(cur_road)가 정확한지, 아니면 old한 정보인지 check 
    항상 맞는 정보: prev, next, lamps_pos(불변)
    아직 업데이트 안되어 있는 정보: road_q, left_q, right_q 
    '''
    length = cur_road.length 
    left_pos = cur_road.start_pos 
    right_pos = cur_road.end_pos 
    left_num = cur_road.left_num 
    right_num = cur_road.right_num 

    if lamps_pos[left_num] == -1 or lamps_pos[right_num] == -1: # 둘 중 하나가 이미 제거된 가로등 
        return False 
    if length == abs(left_pos - lamps_pos[next[left_num]]):
        return True 

def add():
    '''
    인접 가로등 사이에 추가
    '''
    # lazy deletion 
    to_be_broken_road = None 
    while road_q:
        cur_road = road_q[0]
        if valid_check(cur_road):
            # 추가될 기존 길은 road_q에서 삭제 되어야한다. 
            to_be_broken_road = heapq.heappop(road_q)
            break 
        else:
            heapq.heappop(road_q)
    
    # 추가 
    left_pos = to_be_broken_road.start_pos
    right_pos  = to_be_broken_road.end_pos
    left_lamp_num = to_be_broken_road.left_num
    right_lamp_num = to_be_broken_road.right_num

    new_pos = (left_pos + right_pos + 1) // 2 
    new_lamp_num = len(prev)
    heapq.heappush(road_q, Road(abs(new_pos-left_pos), left_pos, new_pos, left_lamp_num, new_lamp_num)) # length, start_pos, end_pos, left_num, right_num
    heapq.heappush(road_q, Road(abs(right_pos-new_pos), new_pos, right_pos, new_lamp_num, right_lamp_num))
    heapq.heappush(left_q, (new_pos, new_lamp_num))
    heapq.heappush(right_q, (-new_pos, new_lamp_num))
    prev.append(left_lamp_num)
    next.append(right_lamp_num)
    prev[right_lamp_num] = new_lamp_num
    next[left_lamp_num] = new_lamp_num

    lamps_pos.append(new_pos)

def remove(removed_lamp_num):
    # 가장 자리 노드가 아닌 중간 노드를 제거하는 경우는, 두 개의 길이 삭제 (lazy deletion)될 것이고
    # 길이 하나 더 추가되어야함. 
    if prev[removed_lamp_num] != -1 and next[removed_lamp_num] != -1:
        length = abs(lamps_pos[prev[removed_lamp_num]]-lamps_pos[next[removed_lamp_num]])
        heapq.heappush(road_q, Road(length, lamps_pos[prev[removed_lamp_num]], lamps_pos[next[removed_lamp_num]], prev[removed_lamp_num], next[removed_lamp_num]))

    '''
    가장 자리 노드가 삭제되면 road_q는 그대로이고, left_q와 right_q도 후에 lazy deletion으로 삭제될 예정이라
    해줄 것이 없음, 다만 현재의 정보를 정확히 lamps_pos, prev, next에 저장
    '''
    # doubly linked list 
    if prev[removed_lamp_num] != -1:
        next[prev[removed_lamp_num]] = next[removed_lamp_num]
    if next[removed_lamp_num] != -1:
        prev[next[removed_lamp_num]] = prev[removed_lamp_num]
    # 현재 가로등에 대한 정보 전부 제거 
    lamps_pos[removed_lamp_num] = -1 
    prev[removed_lamp_num] = -1
    next[removed_lamp_num] = -1 

def get_max_from_left():
    while left_q: # (pos, num)
        (pos, num) = left_q[0]
        if lamps_pos[num] != pos: # invalid 
            heapq.heappop(left_q)
        else:
            break 
    return pos - 1 # r계산 

def get_max_from_right():
    dis = 0
    while right_q:
        (dis, num) = right_q[0]
        dis = dis*-1
        if lamps_pos[num] != dis: # max_heap 이라서 -1 를 곱해줘야함. 
            heapq.heappop(right_q)
        else:
            break 
    return N-dis  # pos는 이미 음수, r 계산 

def get_max_from_roads():
    while road_q:
        cur_road = road_q[0]
        if valid_check(cur_road):
            return cur_road.length / 2
        else:
            heapq.heappop(road_q)

def calculate():
    side_r = max(get_max_from_left(), get_max_from_right())
    middle_r = get_max_from_roads()
    return int(2*max(side_r, middle_r))

lamps_pos = [0]
road_q = [] 
left_q = []
right_q = []
prev = [-1]# 아무것도 없으면 -1 
next = [-1] 
N = 0 

if __name__ == "__main__":
    Q = int(input())
    
    for idx in range(1, Q+1):
        order = list(map(int, input().split()))
        if order[0] == 100:
            N = order[1] 
            check(order[1], order[3:])
        elif order[0] == 200:
            add()
        elif order[0] == 300:
            remove(order[1])
        else:  # 400 
            print(calculate())
```
````

## 코드 트리 채점기 

이런 문제를 푸는 경우, 각 문제의 요구 사항을 만족시키기 위해서는 ***(1) 각 정보의 특성에 맞는 효율적인 자료구조*** 를 설계하고, 각 명령어에 따른 ***(2) 상태 변화를 누락 없이 처리*** 하는 것이 중요하다. 

````{admonition} 필요한 자료구조 
:class: dropdown 

문제를 풀때 wating_urls을 priority queue로 해서, Task를 하나씩 뽑아내는 것을 생각하기 쉽다. 이때 문제는, 현재 들어온 url이 waiting_urls에 있는지 확인하려면, 다 뺐다가 다시 넣어야한다. 따라서, waiting_urls은 검색을 O(1)으로 용이하게 하도록 set[str]으로 만든다. 그렇다면, 우선순위가 가장 높은 Task를 구할때는 domain_pqs: dict[str, list[Task]] 로 각 도메인별로 가장 우선순위가 높은 맨 앞의 Task하나씩 비교해서 보면 된다. 문제의 contraints를 보면 서로 다른 도메인수는 최대 300이라고 했기 때문에, 이렇게 하는것이 시간 복잡도상 가장 효율적이게 된다. 


- `waiting_urls: set[str]`: 채점 대기 큐에 있는 Task들의 url들을 저장하는 집합(set). 특정 url이 큐에 있는지 평균 O(1) 시간 복잡도로 빠르게 확인하기 위해 사용한다. 
- `domain_pqs: dict[str, list[Task]]`: 각 도메인별로 채점 대기 중인 Task들을 저장하는 우선순위 큐. 파이썬의 heapq 모듈을 사용한다. 도메인별로 list가 연결되어있고, 이 리스트가 우선순위큐로 Task들을 저장한다.
    - key는 '도메인(domain)', value는 해당 도메인의 Task들을 담은 우선순위 큐(최소 힙)이다. 
    - Task의 우선순위(우선순위 번호가 작을 수록, 요청 시간이 빠를수록 높음)에 따라 자동으로 정렬되므로, 각 도메인에서 가장 우선순위 높은 Task를 O(1)로 찾을 수 있다. 
- `resting_judger_ids: list[int]`: 쉬고 있는 채점기의 ID들을 저장하는 최소 힙. 가장 번호가 작은 채점기를 빠르게 찾기 위해 사용한다. 
- `judgers: list[Task | None]`: 각 채점기가 현재 어떤 Task를 채점 중인지 저장하는 배열. judger[i]는 i번 채점기가 채점 중인 Task 객체를 가리키거나, 쉬고 있다면 None을 저장한다. 
- `domain_judge_history: dict[str, History]`: 각 도메인의 가장 최근 채점 기록(시작 시간, 종료 시간)을 저장하는 딕셔너리. 채점 유예 기간을 계산하는데 사용한다. 

또한, Task정보와 채점 기록 정보를 편리하게 다루기 위한 클래스들을 정의한다. 

```{code-block} python

# 채점 태스크(Task) 정보를 저장하는 클래스입니다.
class Task:
    def __init__(self, request_time: int, priority: int, url:str):
        domain, pid_str = self.url.split('/')
        self.request_time = request_time 
        self.priority = priority 
        self.url = url 
        self.domain: str = domain 
        self.problem_id: int = int(pid_str)
        self.start_time: int = -1 
        
    
    # 우선순위 비교를 위한 __lt__ 메서드 
    def __lt__(self, other: "Task") -> bool:
        if self.priority != other.priority:
            return self.priority < other.priority 
        return self.request_time < other.request_time

# 각 도메인의 채점 기록(History)을 저장하는 클래스입니다.
# domain_judge_history: dict[str, History]에서 value에 사용될 예정 
class History:
    def __init__(self, start_time: int, end_time: int):
        self.start_time : int = start_time 
        self.end_time : int = end_time 

    def is_valid_time(self, cur_time: int) -> bool:
        gap: int = self.end_time - self.start_time 
        return self.start_time + 3*gap <= cur_time 
```
````