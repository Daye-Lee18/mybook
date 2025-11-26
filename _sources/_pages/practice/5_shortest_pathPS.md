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
- Priority Queue 고득점 Kit 
  - [더 맵게](https://school.programmers.co.kr/learn/courses/30/lessons/42626)
  - [디스크 컨트롤러](https://school.programmers.co.kr/learn/courses/30/lessons/42627)
  - [이중우선순위큐](https://school.programmers.co.kr/learn/courses/30/lessons/42628)

- Priority Queue 
  - [가로등 설치](https://www.codetree.ai/ko/frequent-problems/samsung-sw/problems/street-light-installation/description)
  - [코드 트리 채점기](https://www.codetree.ai/ko/frequent-problems/samsung-sw/problems/codetree-judger/description)
  - [코드트리 투어](https://www.codetree.ai/ko/frequent-problems/samsung-sw/problems/codetree-tour/description)
  - [해적 선장 코디](https://www.codetree.ai/ko/frequent-problems/samsung-sw/problems/pirate-captain-coddy/description)
  - [토끼와 경주](https://www.codetree.ai/ko/frequent-problems/samsung-sw/problems/rabit-and-race/description)

- Dijkstra 
  - [개구리의 여행](https://www.codetree.ai/ko/frequent-problems/samsung-sw/problems/frog-journey/description)
  - [Reachable Nodes in Subdivided Graph](https://leetcode.com/problems/reachable-nodes-in-subdivided-graph/description/?envType=problem-list-v2&envId=shortest-path)
  - [Second Minimum Time to Reach Destination](https://leetcode.com/problems/second-minimum-time-to-reach-destination/description/?envType=problem-list-v2&envId=shortest-path)
  - [Minimum Weighted Subgraph With the Required Paths](https://leetcode.com/problems/minimum-weighted-subgraph-with-the-required-paths/?envType=problem-list-v2&envId=shortest-path)

## Priority Queue, 고득점 Kit 
### 더 맵게

Idea: 
1. 처음 원소의 개수가 짝수이든, 홀수이든 마지막에는 1개의 원소만 남게 된다. 따라서, 계속 진행되었을 때 남아있는 두 개의 원소 중 하나라도 K보다 크면 answer을 내놓으면 됨. 
   1. 또한 answer=0이고, 첫 리스트가 1개만 남아있는 경우도 있으므로, 이를 위해 len(cur_scoville_list) == 1 인 경우를 위해, `while`조건에 맨 처음 원소가 >= K임을 명시한다. 
   2. 따라서, `while`조건에는 K보다 작은 원소들이 존재하는 경우에만 새로 음식을 섞는 작업을 진행한다. 
2. 처음 scoville_pq를 초기화시킬 때, `heappush()` 로 초기화하였기 때문에, 모든 원소가 priority queue의 기준에 따라 정렬되었고 추후에 해당 priority queue에 들어오는 원소들은 당연히 기존 원소들보다 크기 때문에 
    - 제일 작은 원소 1+1*2 = 3 이므로, 원소가 [1, 1, 2] 였다고 하더라도 heapush()로 삽입되기 때문에, 정렬상 작은 원소가 제일 앞에 있게 된다. 
    - for loop으로 heappush로 초기화하면 O(NlogN)이 걸리고, (for loop을 돌면서 heapush())
    - heapify를 하면, O(n)이 걸린다. 배열을 트리 형태로 보고 아래에서 위로 정리 
    - 따라서, python에서는 `heapify()`를 쓰는 것이 훨씬 빠르다. 
  
````{admonition} Solution 
:class: dropdown 

```{code-block} python 
from typing import List
import heapq 

def solution(scoville: List[int], K: int):
    # INIT
    # `scoville` list의 정보를 최소 힙으로 변환, 이것은 아래처럼 O(N)으로 할수도 있고, heapify 함수를 사용해서 할수도 있다. 
    # for idx, each_s in enumerate(scoville):
    #     heapq.heappush(scoville_pq, (each_s, idx))
    #     id_to_scoville.append(each_s)
    heapq.heapify(scoville) # O(N)
    answer = 0 # 연산 횟수 
        
    # 알고리즘 시작 
    while scoville and scoville[0] < K:
        
        # 더 이상 두 개를 섞을 수 없으면 실패 
        # 예) [1,2] -> [5] 로 마지막에는 항상 한 개 만 남음 
        # 예) 애초에 음식이 1개인데, K보다 작은 경우 
        if len(scoville) < 2:
            return -1 
        
        # Step 1. 총 두 개의 음식 꺼내기 
        first = heapq.heappop(scoville)
        second = heapq.heappop(scoville)

        # NOTE: 
        # 1. scoville_pq의 모든 원소는 모두 priority queue형식으로 push되었고
        # 2. 미래에 들어오는 원소들은 항상 클것이므로, 가장 작은 원소가 맨 앞에 있는 것이 진리임
        # 따라서, 혹시 모를 작은 원소때문에 lazy deletion 필요 없음. 
        new_s = first + 2*second 
        heapq.heappush(scoville, new_s) # 새로운 음식의 id = len(id_to_scoville)
        answer += 1 
    return answer # 위의 실패 경우를 제외하면, answer를 반환 


if __name__ == "__main__":
    scoville = [1, 2, 3, 9, 10, 12]; K=7 # 2 
    # scoville = [1, 1]; K=7
    print(solution(scoville=scoville, K=K))
```
````
### 디스크 컨트롤러 

````{admonition} 틀린 정답 
:class: dropdown 

아래 코드의 문제점은 task_pq는 모든 작업을 가지고 있고, 작업의 요청시간이 현재 시간보다 클때에만 해당 task 작업을 진행하게 되어 있음. 
이의 문제점은, 현재 작업이 안된 작업들 중 cur_time보다 request_time 들이 다 높아서, 요청 시간이 가장 작은 task를 꺼내야할 때 문제가 됨. 
task_pq는 요청 시간이 아닌 소요 시간이 가장 짧은 테스크를 min-heap으로 저장해놓았기 때문에 해당 task를 뽑을 수 없음. 

또한, O(N^2log(N))이 걸림. 왜냐하면 `unused_task`를 따로 관리하며, 안 쓰는 것을 넣었다가 빼기 때문임.

-> 이러한 문제점들을 위해서,도착한 작업만 pq에 넣고 + pq가 비면 다음 요청 시각으로 time 점프를 하는 것을 지켜주면 된다. 

```{code-block} python 
from typing import List 
from heapq import heappush, heappop 

class Task:
    def __init__(self, id: int, request_time: int, time_used:int):
        self.id = id 
        self.request_time = request_time
        self.time_used = time_used 
        self.start_time = -1 
        self.end_time = -1 

    def __lt__(self, other):
        if self.time_used == other.time_used:
            return self.id < other.id 
        return self.time_used < other.time_used 
    
    def calculate_turnaround(self):
        assert self.end_time != -1 
        return self.end_time - self.request_time 
    
class History:
    def __init__(self, id=-1):
        self.start_time = -1 
        self.end_time = -1 
        self.task_id = id 
    

def solution(jobs):
    '''
    1 <= jobs.length <= 500
    jobs[i] = [s, l] = [작업 요청 시점, 작업 소요시간]

    최대 N=500, 3NlogN ~ 1e4 
    '''
    # 필요한 자료구조 INIT 
    task_pq = []
    is_harddisk_used = History() # History, task_id being processed 
    task_id_to_object_list = []
    # O(NlogN)
    for id, job in enumerate(jobs):
        cur_task = Task(id, job[0], job[1]) # object sharing 
        heappush(task_pq, cur_task) # jobs.heapify를 하고 싶어도, 단일 정수가 아니라 어려울 듯. 
        task_id_to_object_list.append(cur_task) 
    '''
    아이디어: 
    - pq에 모든 작업을 넣어두고, 매번 "현재 시간까지 도착한 작업들 중에서" 수행할 수 있는 최단 작업을 찾으려고
    pq를 쭉 빼면서 request_time <= cur_time인 task를 만날때까지 탐색 
    - 아직 도착 안 한 것들은 unused_tasks에 모았다가 다시 pqd에 넣기 
    - 시간 복잡도: 최악 O(N^2 log(N))

    
    '''
    cur_time = 0
    total_turnaround_time = 0
    while task_pq: # 최악 N번 
        # 1-1. 하드디스크가 사용 중이면, 작업을 끝냄 / 하드디스크가 사용중이 아님 (맨 처음)
        if is_harddisk_used.task_id != -1 and is_harddisk_used.end_time == -1:
            cur_id = is_harddisk_used.task_id 
            cur_task = task_id_to_object_list[cur_id]
            cur_time = cur_task.start_time + cur_task.time_used 
            # 작업 끝냄 (end time 갱신)
            cur_task.end_time = cur_time 
            is_harddisk_used.end_time = cur_time 
            # 1-2. 작업을 끝냈다면, turnaround time 계산 
            total_turnaround_time += cur_task.calculate_turnaround()

        # 작업을 끝냄과 동시에 다른 작업 시작 가능 
        # 2-1. 현재 시간에서 요청이 된 task 중 (lazy validation) 우선순위가 가장 높은 task 선택
        
        unused_tasks = []
        # O(NlogN)
        while task_pq: # 최대 ~N
            nxt_task = heappop(task_pq) # O(logN)
            if nxt_task.request_time <= cur_time: 
                # 가능 
                # 2-2. 하드디스크가 처리하고 있는 정보 update 
                is_harddisk_used.start_time = cur_time 
                is_harddisk_used.end_time = -1 
                is_harddisk_used.task_id = nxt_task.id 
                task_id_to_object_list[nxt_task.id].start_time = cur_time 
                break
            else:
                #  현재 요청안된 task 
                unused_tasks.append(nxt_task)
        
        # 3. task_pq에서 제거하였으나, 아직 사용안된 task들 다시 넣어주기 
        # O(NlogN)
        for task in unused_tasks:
            heappush(task_pq, task)

        # 현재 시간대에 cur_Time을 위로 늘려줘야만 함. 
        # 도착시간이 가장 빠른 작업을 골라야하는데, task_pq에는 소요 시간이 가장 짧은 작업이 들어있음. 
        if len(unused_tasks) >0 and len(unused_tasks)== len(task_pq):
            cur_task = heappop(task_pq)
            cur_time = cur_task.request_time 
            is_harddisk_used.task_id = cur_task.id
            is_harddisk_used.start_time = cur_time 
            cur_task.start_time = cur_time 
            is_harddisk_used.end_time = -1 
    # while 문에서 마지막 task작업에 대해서 끝마치기 
    if is_harddisk_used.task_id != -1 and is_harddisk_used.end_time == -1:
        cur_id = is_harddisk_used.task_id 
        cur_task = task_id_to_object_list[cur_id]
        cur_time = cur_task.start_time + cur_task.time_used 
        # 작업 끝냄 (end time 갱신)
        cur_task.end_time = cur_time 
        is_harddisk_used.end_time = cur_time 
        # 1-2. 작업을 끝냈다면, turnaround time 계산 
        total_turnaround_time += cur_task.calculate_turnaround()

    
    return int(total_turnaround_time / len(jobs)) # turnaround평균의 "정수 부분"을 출력, 버림
        


    
if __name__ == "__main__":
    # jobs = [[0, 3], [1, 9], [3, 5]] # 8
    # jobs = [[0, 3], [4, 3]] # 8
    jobs = [[0, 10], [3, 1], [3, 2]]# 9
    print(solution(jobs))
```
````

````{admonition} Idea 
:class: dropdown 

```text
'''
- 단 '하나의' 하드디스크를 가지고 있음. 
- 우선순위 디스크 컨트롤러 구현 

1. 자료구조 큐: 작업 요청이 들어왔을 때 (작업 번호, 요청 시각, 작업 소요 시간) 저장
    - 필요한 자료구조 
        - priority queue (task_pq): request_time이 현재 시간보다 작은 작업들을 넣어야함. 
        - jobs.sort()후 현재 시간이 request_time보다 작은 idx를 저장하고 있음. 
2. 하드디스크가 작업을 하고 있지 않고, 대기 큐가 비어있지 않으면 우선순위가 가장 높은 작업을 대기 큐에서 꺼내서 하드디스크에 작업을 시킴
    - 우선순위 
        - 작업의 소요시간이 가장 짧은 것 
        - 작업의 번호가 가장 작은 것 (작업 id는 request_time이 작으면 됨.)
    - 필요한 자료구조 
        - 하드디스크의 작업 여부 is_harddisk_used[harddisk_id] = -1 (False) / True (하드디스크가 작업하는 Task id)
        - Task class (번호, 요청 시각, 작업 소요시간, 작업 시작 시간, end시간, __lt__ 함수  , turnaround 계산 함수 

3. 하드디스크는 작업을 한 번 시작하면 작업을 마칠 때까지 그 작업만 수행 
    - 필요한 자료구조 
        - 하드디스크 id - History class (task_id, 시작 시간, end 시간)

4. 하드디스크가 어떤 작업을 마치는 시점과 다른 작업 요청이 들어오는 시점이 "겹치면" 하드디스크가 작업을 마치자마자 디스크 컨트롤러는 요청이 들어온
작업을 "대기 큐"에 저장한 뒤 우선순위가 높은 작업을 대기 큐에서 꺼내서 하드디스크에 그 작업을 시킨다. 
- 요청을 한 작업이 들어오고 그 같은 시점에 하드디스크가 작업이 끝나면 바로 다른 작업을 시작할 수 있다. 
    - 하드디스크가 작업을 하고 있는 중이면, -> 끝나는 지점 history에 확인해서 반환시간 계산하기  
    - 작업이 끝나면 다음에 할 작업이 있는지 대기 큐에서 받으면 됨. 

5. 하드디스크가 어떤 작업을 마치는 '시점'에 다른 작업이 들어오지 않더라도 그 작업을 마치자마자 또 다른 작업을 시작할 수 있다. 
- 이 과정에서 걸리는 시간은 없다고 가정한다. 

Algorithm 
0. 필요한 자료구조 
- Task class, History class, task_pq 생성, 
- is_harddisk_used[harddisk_id] = -1 (False) / True (하드디스크가 작업하는 Task id), 
- 대기 큐 Task class에 한번에 저장해놓기 
- History class (task_id, 시작 시간, end 시간)

Algorithm 
    1. 현재 수행된 작업 개수(cnt) n이 원래 작업 개수보다 작은 경우에 아래 스텝을 계속 진행 
    2. 현재 jobs.sort()[idx] request_time <= cur_time보다 작은 것들을 pq에 넣음 
    3. pq에서 가장 작은 것을 뺌 
        3-1. 만약 pq에 원소가 없다면, cur_time을 jobs.sort()에서 현재 idx.request_time의 값으로 대체 
    4. turnaround_time은 cur_time + required_time (소요시간) 이고 cur_time += required_time으로 업데이트 
'''

import math 
N = 500 

print(N*math.log(N)) # ~1e4 
```
````
````{admonition} Solution 
:class: dropdown 

```{code-block} python 
import heapq 

def solution(jobs):
    # 1. 요청 시간 기준으로 정렬 
    jobs.sort(key=lambda x: x[0]) # O(NlogN)

    # 필요한 자료 구조 
    heap = [] # (작업 소요 시간, 요청 시간)
    time = 0 # 현재 시각 
    idx = 0  # jobs에서 아직 힙에 안 넣은 인덱스 
    count = 0 # 처리한 작업 수 
    total_time = 0 # 요청 ~ 완료 시간 합 

    n = len(jobs)

    while count < n:
        # 2. 현재 시각까지 들어온 모든 job을 heap에 넣기 
        '''
        각 job은 딱 한 번 heap에 push 되고 (idx가 전역에서 위로만 움직임), 딱 한 번 pop 됨
        push, pop 각각 O(log N), N개에 대해 한 번씩 → O(N log N)
        '''
        while idx < n and jobs[idx][0] <= time:
            req, dur = jobs[idx]
            heapq.heappush(heap, (dur, req))
            idx += 1 


        if heap:
            # 3. 가장 작업 시간이 짧은 job을 처리 
            dur, req = heapq.heappop(heap)
            time += dur  # 이 작업이 끝나는 시각 = 현재 시각 
            total_time += time-req # 요청 ~ 완료 시간 (turnaround)
            count += 1 
        else:
            time = jobs[idx][0]

    return total_time // n
    
if __name__ == "__main__":
    # jobs = [[0, 3], [1, 9], [3, 5]] # 8
    # jobs = [[0, 3], [4, 3]] # 8
    jobs = [[0, 10], [3, 1], [3, 2]]# 9
    print(solution(jobs))
```
````

### 이중우선순위큐 

````{admonition} Explanation 
:class: dropdown 

```text
이중 우선순위큐는 3가지 연산을 할 수 있다. 
1. "I 숫자" 
    - 큐에 주어진 숫자를 삽입 
    - 숫자는 음수일 수 있음. 
2. D 1 
    - 큐에서 최댓값 삭제 
    - 빈 큐에 데이터를 삭제하라는 연산이 주어질 경우, 해당 연산은 무시 
3. D -1 
    - 큐에서 최솟값 삭제 
    - 빈 큐에 데이터를 삭제하라는 연산이 주어질 경우, 해당 연산은 무시 


매개변수로 이중 우선순위 큐가 할 연산 operations이 주어질 때, 모든 연산을 처리한 후 큐가 비어있으면 [0,0], 
비어있지 않으면 [최댓값, 최솟값]을 return 하도록 solution 함수를 구현해라. 

1 <= operations.length <= 1_000_000 
```
````

````{admonition} Solution
:class: dropdown 

만약, visited 리스트를 만들어서 False/True로 지연 삭제를 만들고 싶은 경우에는 
visited = [False] * len(operations)를 하고 
visited[i] = True 로 한다. 이때 id 숫자는 음수일 수 있으므로, id는 operation index (0, 1, 2, ...)로 만들어준다. 

```{code-block} python 
from heapq import heappush, heappop 

# 필요한 자료구조 
min_heap = []
max_heap = []
does_id_exist = set()  # True/False 

def solution(operations):
    global min_heap, max_heap, does_id_exist
    for op in operations:
        cmd = op.split()
        if cmd[0] == "I":
            heappush(min_heap, int(cmd[1]))
            heappush(max_heap, -1*(int(cmd[1])))
            does_id_exist.add(int(cmd[1]))
        
        # 최댓값 삭제 
        elif cmd[0] == "D" and cmd[1] == "1":
            # lazy deletion 
            while max_heap:
                cur_num = heappop(max_heap) * -1 
                if cur_num in does_id_exist:
                    does_id_exist.remove(cur_num)
                    break
        
        # 최솟값 삭제 
        else:
            # lazy deletion 
            while min_heap:
                cur_num = heappop(min_heap)
                if cur_num in does_id_exist:
                    does_id_exist.remove(cur_num)
                    break 
    # 모든 연산 처리후 가장 최신 업데이트에는 does_id_exist가 가지고 있음 
    # 큐가 비어있으면 [0,0]
    # 비어있지 않으면, [최댓값, 최솟값]을 return 
    does_id_exist = list(does_id_exist)
    does_id_exist.sort() # ascending sort 
    return [does_id_exist[-1], does_id_exist[0]] if does_id_exist else [0, 0]

# operations = ["I 16", "I -5643", "D -1", "D 1", "D 1", "I 123", "D -1"] # [0,0]
operations = ["I -45", "I 653", "D 1", "I -642", "I 45", "I 97", "D 1", "D -1", "I 333"] # [333, -45]
print(solution(operations))
```
````

## Priority Queue 

우선순위에 대해서 맨 처음의 데이터를 Extract하거나 데이터의 출입이 잦은 경우, priority queue를 이용해 데이터를 관리한다. 

### 가로등 설치 


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

### 코드 트리 채점기 

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
    '''
    "Task"라고 표시한 이유: __lt__가 클래스 내부에 있을 때 아직 정의되지 않은 Task 타입을 참조하기 위해 "전방 선언(Forward Reference)"으로 문자열을 사용한 것. 즉 문자열을 '타입 힌트'로 쓰면 python이 나중에 실제 그걸 클래스 이름으로 다시 해석한다. 
    '''
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

````{admonition} solution
:class: dropdown 

```{code-block} python
import sys
import heapq 
from collections import defaultdict 

# -----------------------------
# 데이터 구조 공간 복잡도
# -----------------------------
# domain_pqs           : 도메인별 우선순위 큐, 전체 Task 수 ≤ Q → O(Q)
# domain_judge_history : 도메인 ≤ 300개 → O(1)
# resting_judger_ids   : 최대 N개 → O(N)
# judging_domains      : 최대 300개 → O(1)
# judgers              : N+1 크기 배열 → O(N)
# waiting_urls         : 대기 URL ≤ Q → O(Q)
#
# 전체 공간 복잡도: O(N + Q)
# -----------------------------


class Task:
    """Task 객체 1개 공간: O(1) (URL 길이가 상수 19니까 O(1))"""
    def __init__(self, request_time: int, priority: int, url:str) -> None:
        domain, pid_str = url.split('/')

        self.request_time: int = request_time
        self.start_time: int = -1 
        self.priority: int = priority 
        self.url: str = url 
        self.domain: str = domain 
        self.problem_id : int = int(pid_str)

    def __lt__(self, other: "Task") -> bool:
        """비교 연산: O(1)"""
        if self.priority != other.priority:
            return self.priority < other.priority 
        return self.request_time < other.request_time 


class History:
    """History 객체 1개 공간: O(1)"""
    def __init__(self, start_time: int, end_time: int)-> None:
        self.start_time: int = start_time
        self.end_time: int = end_time

    def is_valid_time(self, cur_time: int) -> bool:
        """단순 연산: O(1)"""
        gap: int = self.end_time - self.start_time 
        return self.start_time + 3 * gap <= cur_time 
    

# 전역 데이터 구조
n: int = 0
domain_pqs: dict[str, list[Task]] = defaultdict(list)
domain_judge_history: dict[str, History] = dict()
resting_judger_ids: list[int] = []
judging_domains: set[str] = set()
judgers: list[Task | None] = []
waiting_urls :set[str] = set()


# ----------------------------------------------------------
# 명령어 100: 채점기 준비
# ----------------------------------------------------------
# 시간 복잡도:
#   resting_judger_ids 초기화: O(N)
#   judgers 초기화: O(N)
#   process_200 호출: O(log Q)
# 전체 → O(N)
#
# 공간 복잡도: O(N) (리스트 초기화)
# ----------------------------------------------------------
def process_100(n_: int, u:str) -> None:
    global n, resting_judger_ids, judgers 

    n = n_
    resting_judger_ids = [i for i in range(1, n+1)]  # O(N)
    judgers = [None] * (n+1)                         # O(N)

    process_200(t=0, p=1, u=u)                       # O(log Q)


# ----------------------------------------------------------
# 명령어 200: 채점 요청 추가
# ----------------------------------------------------------
# 시간 복잡도:
#   u in waiting_urls (set membership): O(1)
#   Task 생성: O(1)
#   heapq.heappush: O(log Q)
#   waiting_urls.add: O(1)
# 전체 → O(log Q)
#
# 공간 복잡도: O(1) (Task 1개 추가 → 전체적으로 O(Q) 안에 포함)
# ----------------------------------------------------------
def process_200(t:int, p: int, u: str) -> None:
    global waiting_urls, domain_pqs 

    if u in waiting_urls:    # O(1)
        return 
    
    task = Task(request_time=t, priority=p, url=u)  # O(1)
    heapq.heappush(domain_pqs[task.domain], task)   # O(log Q)
    waiting_urls.add(u)                             # O(1)


# ----------------------------------------------------------
# 도메인 채점 가능 여부 확인
# ----------------------------------------------------------
# 시간 복잡도: O(1)
# 공간 복잡도: O(1)
# ----------------------------------------------------------
def is_domain_judgeable(cur_time: int, domain: str) -> bool:
    global judging_domains, domain_judge_history

    if domain in judging_domains:           # O(1)
        return False 

    history: History | None = domain_judge_history.get(domain, None)  # O(1)
    if history and not history.is_valid_time(cur_time):               # O(1)
        return False 
    
    return True 


# ----------------------------------------------------------
# 명령어 300: 채점 시도
# ----------------------------------------------------------
# 시간 복잡도:
#   쉬는 채점기 체크: O(1)
#   모든 도메인 순회: D ≤ 300 → O(300) = O(1)
#   각 도메인 pq[0] 접근: O(1)
#   비교 연산: O(1)
#   heappop(resting_judger_ids): O(log N)
#   heappop(domain_pqs[domain]): O(log Q)
#   set/dict 업데이트: O(1)
#
# 전체 → O(log N + log Q)
#
# 공간 복잡도: O(1)
# ----------------------------------------------------------
def process_300(t: int) -> None:
    global resting_judger_ids, domain_pqs, judging_domains, judgers, waiting_urls

    if not resting_judger_ids:  # O(1)
        return 
    
    best_task: Task | None = None 

    for domain, pq in domain_pqs.items():   # O(300) = O(1)
        if not pq or not is_domain_judgeable(cur_time=t, domain=domain):  
            continue 

        current_task = pq[0]                # O(1)
        if not best_task or current_task < best_task:  # O(1)
            best_task = current_task 

    if not best_task:
        return 
    
    j_id: int = heapq.heappop(resting_judger_ids)     # O(log N)
    best_task.start_time = t 

    heapq.heappop(domain_pqs[best_task.domain])       # O(log Q)
    judging_domains.add(best_task.domain)             # O(1)
    judgers[j_id] = best_task                         # O(1)
    waiting_urls.remove(best_task.url)                # O(1)


# ----------------------------------------------------------
# 명령어 400: 채점 종료
# ----------------------------------------------------------
# 시간 복잡도:
#   task 조회: O(1)
#   History 생성 및 dict 기록: O(1)
#   heapq.heappush(resting_judger_ids): O(log N)
#   set.remove + 배열 저장: O(1)
#
# 전체 → O(log N)
#
# 공간 복잡도: O(1)  (History 1개 추가 → 전체 O(Q)에 포함)
# ----------------------------------------------------------
def process_400(t: int, j_id: int) -> None:
    global judgers, domain_judge_history, resting_judger_ids, judging_domains

    task: Task | None = judgers[j_id]       # O(1)
    if not task:
        return

    domain_judge_history[task.domain] = History(start_time=task.start_time, end_time=t)  # O(1)
    
    heapq.heappush(resting_judger_ids, j_id)    # O(log N)
    judging_domains.remove(task.domain)          # O(1)
    judgers[j_id] = None                         # O(1)


# ----------------------------------------------------------
# 명령어 500: 대기 큐 조회
# ----------------------------------------------------------
# 시간 복잡도: O(1)
# 공간 복잡도: O(1)
# ----------------------------------------------------------
def process_500(t: int) -> None:
    global waiting_urls
    print(len(waiting_urls))  # O(1)


# ------------------ 메인 실행 ------------------

'''
전체 시간 복잡도: O(Q(logN + logQ))
전체 공간 복잡도: O(N+Q)
'''
q = int(input())
for _ in range(q):
    query: list[str] = input().split()
    cmd: int = int(query[0])

    if cmd == 100:
        process_100(n_=int(query[1]), u=query[2])
    elif cmd == 200:
        process_200(t=int(query[1]), p=int(query[2]), u=query[3])
    elif cmd == 300:
        process_300(t=int(query[1]))
    elif cmd == 400:
        process_400(t=int(query[1]), j_id=int(query[2]))
    elif cmd == 500:
        process_500(t=int(query[1]))


```
````

### 코드 트리 투어 

구현 문제의 경우, 실수할 수 있는 부분은 여러개의 command 중 하나를 하고 다른 command를 할 경우 그 다음 상황에 영향을 미치는 것이 있는 지 미리 체크해야한다. 이번 문제의 경우 새로 변한 start_id에 대해 이후에 command `200`이 나오는 경우에 새로운 `start_id`에 대하여 Trip의 cost등을 계산해야하므로 이를 global variable로 관리해주는 것이 포인트였다. 

````{admonition} explanation 
:class: dropdown 

```{code-block} python
'''
여행 상품의 출발지는 통일: 0번 도시 
간선에 음수가 없으므로 dijkstra algorithm사용 가능. 

0. 필요한 자료 구조 
- `trip_lists`: pq, Trip class를 priority에 맞게 보관 
- class Trip: __lt__(), get_benefit(), 
- shortest_path: dict[int(id), list[int] (cost to the dest_id)]
- graph: adjacency_list: list[list[int]], 각 노드와 연결된 node정보 저장 

1. 코드트리 랜드 건설
- 도시의 수 n, 간선의 수 m, 간성 정보 (도시 i,도시 j, 가중치)
- 자기자신 연결, 두 도시 간의 여러 간선 가능 
- start_node가 0인 shortest_path INIT 
- graph 초기화 

2. 여행 상품 생성 
- 관리 목록: (id, revenue_id, dest_id) 여행 상품을 추가하고 관리 목록에 추가 
    - 데이터 구조: priority queue로 O(log(30,000)) = O(10)  
    - id: 여행상품 고유 식별자 
    - revenue_id : 여행사가 얻게되는 매출 
    - start_id: 이 상품의 출발지 
    - dest_id :이 상품의 도착지 
    - cost : revenue_id - shortest_path (출발지 ~ 도착지 최단거리)
- 해당 명령은 최대 30,000번 주어진다.
- class Trip 생성으로 해당 데이터 관리 필요 
    - __lt__ 함수 : pq에 들어갈나 <, >를 계산할때 비교 함수 생성 
    - get_benefit 함수: Start_id가 바뀜에 cost가 변해서, 변할때 그 차이를 계산해주는 함수 생성 


3. 여행 상품 취소 
- 여행 상품 고유 식별자 id에 해당하는 여행 상품 존재하면 관리 목록에서 삭제 
    - 찾기: O(1) -> dict로 id에 따른 상품을 관리할 수 있음 
        - list로 관리하면 어떤 id가 있을 지도 모르는데, 메모리 비용이 많이 든다. 
        - id는 30,000보다 작으므로 근데 어차피 여행 상품 넣는 것을 30000으로 하니까 오히려 list의 indexf로 관리하는게 더 나을 수도 있겠음. 
        - dictionary는 key가 맞는지 일일이 확인해야하잖아. 
    - 삭제: O(1) 
    - 삭제 이후, 관리 목록 pq 는 lazy deletion을 진행해야함. (즉, id_trip_pq는 바로 관리하고, 4번 판매할때, 이게 유효한지, id_trip_pq를 통해서 확인)
- 해당 명령은 최대 30,000번 주어진다. ~ 3*1e4

4. 최적 여행 상품 판매 
- 판매 가능 상품이 없으면 -1 출력후 상품 제거 x 
- 판매 가능 상품 있으면 해당 상품 id 출력후 상품 제거 
- 해당 명령은 최대 30,000번 주어진다. ~ O(10)*30000 ~ 3*1e5

- 판매 불가 상품 
    - 출발지로부터 dest_id까지 도달하는 것이 불가능 
        - 이건 넣을때부터 알 수 있는 건데...아 근데 넣을때 빼버리면 나중에 출발지 변경할때 또 달라질 수 있어서 그때그때 검사하는 게 좋음 
        - Trip class의 cost가 MAX값이면 도달 불가능확인 가능 
        - O(1)
    - cost_id가 revenue_id보다 커서 여행사가 이득을 얻을 수 없는 상황, 이득이 0인 경우도 팔 수 있음 
        - revenue_id - cost_id < 0 이면 팔 수 없음 
        - O(1) 
- 관리 목록에서 조건에 맞는 최적 상품 선택 및 판매 
    - 조건: 
        - 이득 [revenue - cost]가 최대인 상품 : cost = 출발지로부터 id 상품의 도착지 dest_id까지 도달하기 위한 최단 거리 
        - 이득이 동일하면 id가 가장 작은 상품 선택 
        - NOTE: 
        ---> 이 조건들을 생각해보면, get_benefit()을 통해 나온 이득이 음수인 경우, 그 뒤를 보지 않아도 다 팔리지 못할 상품들이므로 팔 수 있는 상품이 없다고 생각하면 됨. 
        ----> 따라서, 맨 처음 pq에 있는 상품의 이득이 >=0인 경우, 그것을 팔면 된다.
    - 판매 불가 조건 
        - 위 참고 
    - 판매 가능 상품 중 가장 우선순위가 높은 상품을 1개 판매하고, 이 상품의 id를 출력한 뒤 관리 목록에서 제거 (logQ)
        -관리 목록: priority queue 

Algorithm 
1-1. pq에서 맨 위의 Task를 뽑는다. (아직 제거x) (이득이 최대이고, id가 가장 작은 상품 순으로 뽑힘.)
2-1. 이미 3.에 의해 삭제된 여행 상품인지 확인한다. (O(1))
2-2. 존재하는 상품이면, 판매 불가 상품인지 확인한다. (O(1))
3-1. 위의 조건을 만족하면, id를 출력한 뒤 관리 목록에서 제거한다. (log30000 ~ 10)
3-3. pq의 모든 상품을 다 돌았는데도 불구하고, 판매가능한 상품이 없으면, -1를 출력하고, 관리 목록에서 제거하지 않는다. 


5. 여행 상품의 출발지 변경 
- 여행 상품의 출발지를 전부 s로 변경 -> 변경 이후에 각 상품의 cost_id가 변경될 수 있음에 유의 
- NOTE: `trip_lists`의 모든 Trip에 대해서 start_id와 cost를 변경해서 pq안에서 정렬이 바뀔 수 있도록 해야한다. 
    - 새로 삽입될 때 비교를 해서 넣고, 해당 class안의 variable들을 바꿨다고 우선순위가 변할 것 같지 않음 
    - 새로 만들때 log(30000) = O(10)
    - 새로 만들고 기존 변수명에 다시 복사하는 방식을 사용해야함. global 및 [:] 슬라이싱사용. 
- 출발지가 바뀌는 경우, cost를 쉽게 계산하는 법 
    - dijkstra는 greedy algorithm으로 O(E)로 계산 가능 : 이걸로 최대 15번이니까 15*O(Elog(V))
    - floyd warshall algorithm은 한번에 2D matrix 계산 O(N^3)이 걸림 
- NOTE: 새로 변한 start_id에 대해 이후에 command `200`이 나오는 경우에 새로운 `start_id`에 대하여 Trip의 cost등을 계산해야하므로 이를 global variable로 관리해준다. 

- 해당 명령은 최대 15번 주어진다. 0<=s<=n-1 인데, n은 최대 2000개지만, s는 15번만 바뀐다. 
- s가 바뀌면 그에 따른 최단거리도 달라진다. -> Trip의 cost , start_id 가 변해야함. 
- shortest_distance: dict[int, list[int]] 를 저장하는 데이터 구조 필요 -> Trip의 cost는 shortest_distance[start_id][dest_id]로 저장 
- Dijkstra algorithm을 최대 15번 진행한다고 하면, O(15*(2000+10000)* log(2000)) = 1.3 * 1e6
'''
import math 
print(12000*15*math.log(2000))
```
````

````{admonition} solution 
:class: dropdown 

만약 풀리지 않는 test case가 있다면 각 cmd마다 print를 해서 어느 부분이 틀렸는지 검토해보자. 

```{code-block} python 
import sys 
import heapq 

input = sys.stdin.readline
# sys.stdin = open('Input.txt')

def graph_init(num_edges: int, edges: list[int]):
    global graph 
    for idx_2 in range(2, len(edges), 3):
        # print(idx)
        node1, node2, weight=edges[idx_2-2], edges[idx_2-1], edges[idx_2]
        # 자기자신 연결인 경우는 한번만 그래프에 추가 
        graph[node1].append((weight, node2))
        if node1 != node2:
            graph[node2].append((weight, node1))


def dijkstra(start_node):
    global shortest_path, graph, N
    # shortest_path: dict[int, list] INIT 
    shortest_path[start_node] = [MAX]* N
    shortest_path[start_node][start_node] = 0 # INIT at start_node 
    q = [(0, start_node)] # (dis, start_node)

    while q:
        cur_dis, cur_node =  heapq.heappop(q) # min_heap on the first dis element 

        # 1. is already visited?
        if cur_dis > shortest_path[start_node][cur_node]:
            continue 
        
        # 2. explore next nodes in the graph
        for weight, nxt_node in graph[cur_node]:
            dis = cur_dis + weight
            if  dis < shortest_path[start_node][nxt_node]:
                shortest_path[start_node][nxt_node] = dis 
                heapq.heappush(q, (dis, nxt_node))
    # print(shortest_path)
def process_400() -> None:
    global trip_pq

    while trip_pq:
        cur_trip = trip_pq[0]

        # 이미 cmd[0]== 300에 의해 삭제된 여행 상품인지 확인
        # lazy deletion  
        if not exist_trip_id[cur_trip.id]:
            heapq.heappop(trip_pq)
            continue 

        # 존재하는 상품이면, 판매 불가 상품인지 확인 
        if cur_trip.get_cost() == MAX or cur_trip.get_benefit() < 0:
            print(-1)
            return 
        else:
            # 위의 조건을 다 만족하면, id를 출력한 뒤 관리 목록에서 제거 
            print(cur_trip.id)
            heapq.heappop(trip_pq)
            # NOTE: 해당 관리 목록에서 삭제해줘야함!!
            exist_trip_id[cur_trip.id] = False
            return 
    
    # trip_pq에 아무원소도 없을 경우 
    print(-1)

        
# --------------- 전역 스코프 
Q = int(input())

# 데이터 구조 INIT 
MAX = int(1e9)
shortest_path: dict[int, list[int]] = dict()
trip_pq: list["Trip"] = [] 
exist_trip_id: list[bool] = []
cur_start_id: int = 0

# graph 는 아래서 
class Trip:
    def __init__(self, id:int, revenue: int, dest_id: int, start_id: int=0):
        self.id = id 
        self.revenue = revenue 
        self.dest_id = dest_id 
        self.start_id = start_id # 변할 수 있음. 
        # self.cost = MAX  # start 와 dest을 알면 shortest_path에 바로 접근해서 알 수 있고, 아래 get_benefit() 함수에서 그걸 구현해놓음. 
    
    def __lt__(self, other):
        my_b = self.get_benefit()
        other_b = other.get_benefit()
        if my_b == other_b:
            return self.id < other.id 
        return my_b > other_b 
    
    def get_benefit(self): # Revenue - current_cost 
        global shortest_path
        assert self.start_id in shortest_path 
        return self.revenue - shortest_path[self.start_id][self.dest_id]
    
    def get_cost(self):
        global shortest_path
        assert self.start_id in shortest_path 
        return shortest_path[self.start_id][self.dest_id]

    def __repr__(self):
        return f"Trip({self.id})"
    
for _ in range(Q):
    cmd = list(map(int, input().split()))
    # if cmd == [300, 3]:
    #     print('a')
    if cmd[0] == 100:
        N = cmd[1]; M=cmd[2]
        graph = [[] for _ in range(N)]
        exist_trip_id = [False] * 30001
        graph_init(M, cmd[3:]) # m, [node1, end1, weight1,... ]
        # print(graph)
        dijkstra(0) # start node= 0 
        # print(shortest_path)

    elif cmd[0] == 200:
        heapq.heappush(trip_pq, Trip(id=cmd[1], revenue=cmd[2], dest_id=cmd[3], start_id=cur_start_id))
        exist_trip_id[cmd[1]] = True  # 해당 여행이 존재 
        # print(f"Add Trip {cmd[1]}")
        # print(f"  trip_pq: {trip_pq}")
        # print(f"  exist_trip: {exist_trip_id[cmd[1]]}")

    elif cmd[0] == 300:
        cur_id = cmd[1]
        if exist_trip_id[cur_id]: # 존재하면 
            exist_trip_id[cur_id] = False 

        # print(f"Delete Trip {cur_id}")
        # print(f"  trip_pq: {trip_pq}")
        # print(f"  exist_trip: {exist_trip_id[cur_id]}")


    # 최적의 여행 상품 판매 
    elif cmd[0] == 400:
        # print(f"여행 상품 판매")
        process_400()
        # print(f"  trip_pq: {trip_pq}")



    # 여행 상품의 출발지 변경 
    else: # cmd[0] == 500:
        # print(f"-------출발지 변경")
        # print(f"before: {trip_pq}")
        ##### global로 start_id 를 변경해줘야 나중에 200으로 들어왔을 때, 해당 start_id에 대해 task가 정렬됨. 
        cur_start_id = cmd[1]
        new_trip_pq = []

        # 1. 새로운 출발점에 대한 shortest_path 계산 먼저 해야 cost 계산에 의해 heapq에서 정렬됨. 
        dijkstra(cur_start_id)

        for cur_trip in trip_pq:
            # 여기서도 lazy deletion해준다. 
            if not exist_trip_id[cur_trip.id]:
                continue 
            cur_trip.start_id = cur_start_id
            heapq.heappush(new_trip_pq, cur_trip)

        trip_pq = new_trip_pq[:]
        # debug 
        # print(f"start node {cur_start_id}: {trip_pq}")
        # print([exist_trip_id[cur_trip.id] for cur_trip in trip_pq])
        # print(f"-------출발지 변경 끝")
```
````

### 해적 선장 코디 

해당 문제는 제일 우선순위가 높은 1개를 뽑는 것이 아닌, 조건에 맞는 최대 5개의 선박을 고를 수 있다는 것에 있다. 

````{admonition} explanation 
:class: dropdown 

```{code-block} python
'''
- '코디'라는 이름의 선장이 `대형 선박` 1개을 침몰시키려고 한다. 
- 총 T개의 명령 
- '코디'는 여러척의 공격할 수 있는 선박을 가지고 있다. 
- 각 명령은 1시간 단위로 실행된다. i번째 명령이 수행된 뒤, 1시간이 지나면 i+1번째 명령이 수행된다. 


필요한 자료 구조 
- class Ship 
    - __lt__ : sort()에서 key에 사용될 함수 필요 이거 __lt__로 구현해도 되는건가?
    - __init__: id, 재장전 시간 (r), 공격력 (p), status = 0 
    - 다른 변수들: 공격에 사용된 시간 (used_time)
    - 현재 시간이 t인데, abs(used_time - t) >= r이면 status를 0으로 바꾸어야함. 
- ship_pq: priority queue 최우선 선박순으로 나열
    - status 
    - pw
    - id 
- id_to_ship_dict: cmd 300때 Id에 따라서 선박에 바로 접근할 수 있도록 함. 
    - object sharing 필요 (어디에서 해당 object안의 variable값을 바꾸어도 나중에 접근할때도 동일한 값이도록)
- used_ship_list: 
    - cmd 400의 맨 위에서 재정비할 ship들이 있는 지 분석 

각 cmd뒤에 다른 cmd를 할 때 영향을 주는 것이 있는 지 확인 

아래의 명령의 순서를 확인했을 때, cmd 400이 들어오기 전에 `ship_lists`는 반드시 정렬되어 있어야한다. 

- 200 -> 200: 지원 요청 (`id_to_ship_dict`와 `ship_lists`에 추가) -> 지원 요청  
- 200 -> 300: 지원 요청한 후, 함포 교체 `id_to_ship_dict` 에서 object안의 변수 바꾸면 됨. 
- 200 -> 400: 지원 요청 -> 공격 명령
    - 공격 명령 전에, sort()안에서 지원 요청때도 다시 sort()가 필요함. 그럼 평균 O(30000log30000) ~ 3 * 1e5 걸림. 
    - queue:
        - 지원 요청 200: O(logN)
        - 공격: 최대 5개인데, O(logN) 을 최대 N개 해야할 수도 있다. 
        - 교체 후 다시 넣기 : O(logN), lazy check해야함. 나중에 공격전에
    - sort():
        - 지원 요청시 삽입 O(1)
        - 공격: sort()를 진행 NlogN -> 최적의 Ship찾기 (O(1) 이도 마찬가지로 dictionary에 접근하여 list안의 원소를 바꿔준후, 다시 sort()를 하면 NlogN이 된다. 


- 300 -> 200: 교체 후 지원 요청 
    - 교체하고, 하나 추가하면 order가 바뀌어야하는데...? 
- 300 -> 300: 교체 후 또 교체  
- 300 -> 400: 교체 후 공격 명령
    - queue: 문제는 어떤 Ship이 1 -> 100000으로 공격력이 높아졌을 때, 다시 안 넣으면 우선순위 큐 맨 앞에 있지 않아서 다시 넣어야하고
        - NOTE: 이때 dictionary에 있는 Ship을 교체하면 sort() list안에 있는 객체도 바뀐다.
        - 다만, 객체의 정보가 바뀐다고 해서 queue안에 정렬순서가 바뀌지는 않게된다. 따라서, 새로운 객체를 다시 삽입 (logN)하고 
        - NOTE: 만약 공격 명령때 같은 id가 나오면 삭제한다. (logN)
        - 최대 5개를 뽑아야하지만, 최대 N=3*1e4라고 하면, 5000* NlogN (최대 N개에 대해서 pop을 함, 이 cmd는 최대 5000번이 주어짐)
            -> 2*1e9 NOTE: 시간초과 
        - 그러면, 맨 앞의 원소들은 status=0이며, 우선순위가 높은 것으로 하면 5000 * NlogN이 아니라, 5000*5*logN으로 줄어든다. 

    - sort:
        - 교체를 한후
        - 400 function안에서 sort()를 진행 NlogN 최대 5000번까지 주어짐. -> N이 최대 3*1e4라고 하면,  2*1e9 NOTE: 시간초과 
        - 또한 sort()후 각 인덱스에서 최대 N=3*1e4개 중 status가 0인 선박을 찾고 등등 해야함. 
        -> 더 많이 걸림 


- 400 -> 200: 공격 후 지원 요청 
- 400 -> 300: 공격 후 교체 (공격력만 교체)
- 400 -> 400: 공격 후 공격 
    - 생각해보니, 공격을 한 후에는 재점검 시간이 필요해서 다시 공격할 수 없음 
    - stauts를 재정비 (1)로 바꾸고, 이경우에는 사용할 수 없도록 해야함. 
        - 이 때, status=1이지만, current_time - 공격시간 >= r 이면 공격 가능함. 
    - 따러 used_ships을 설정하면 시간 관리도 해야해서 로직이 복잡해짐. 재정비 시간이 지나면 다시 status를 하나하나 바꿔야해서 복잡해짐. 
    - 공격때 사용한 ship들을 따로 list로 최대 len(5)로 관리하고, 
    - 공격 때 queue에서 pop한 다음, 다시 Status를 1로 해서 insert한다. 
    - 다만, 재정비시간이 1인 경우는 다음에 바로 사용할 수 있으며, 
    - 또 공격이 계속 연달아 있는 경우에 대비하여, status들을 바로 바꿔줘야하는데, 이때 재정비 시간이 지나서 다시 공격할 수 있는 상황에서는 
        - status로 관리하게 되면 불편한 경우가 있다. 
    - queue를 __lt__를 
        - status 
        - 공격력 , Id 
    - 공격때 사용된 ships들은 5*5000번 => 최대 3e4개일 수 있는데, 
        - 이를 하나하나 돌아가면서, 재정비하면 status를 바꾸고 queue에 다시 삽입하면, O(3e4* log(4e4)) ~ 3e5 정도의 시간이 걸린다. 
    

1. 공격 준비 (100 N id_1 p_1 r_1 id_2 p_2 r_2 ...... id_N p_N r_N)
- '코디'의 N척의 선박에 사격 준비를 지시 
- 각 ship은 id, power (공격력, p), 재장전 시간 (r), status (초기는 모두 사격대기)를 가짐 
    - status: 사격 대기 (INIT, 0) -> 공격 후 재장전 1  (r시간이 경과하면 사격대기로 전환)
        - lazy check 해야할 것 같음. 
    - 이를 위한 class Ship 이 필요함 

2. 지원 요청 (200 id p r)
- 추가 병력을 요청하여 1개의 새로운 선박이 합류됨 
- 새로 합류한 선반 (status = 사격 대기 0, 선박 번호 id, 공격력 p, 재장선 시간 r)
- 이 명령은 최대 30,000번까지 주어짐.
- id_to_ships와 ship_list 두개 다에 삽입 

3. 함포 교체 (300 id pw)
- 이 명령은 최대 10,000번까지 주어짐. 존재하지 않는 선박 번호 id가 주어지지 않음. 
- id번의 선박의 함포를 교체 
    - id번의 선박에 바로 접근할 수 있어야함. 
    - 문제는 1 <= id <= 1e9, 너무큰데...? memory limit이 128 MB인데, int가 4byte이면 32*10^6개의 밖에 list형성 불가 
    - 이때 명령의 개수가 최대 50,000개 이므로 dictionary로 저장하면, 2*1e5개 저장, Ship class도 하나당 9 byte라고 하면, 4.5 * 1e5이므로 
    - 일단 `id_to_ship_dict: dict[int, Ship]`으로 만듦. 
    - 이때 id를 찾을 때 time complexity는 string이 아니라 int라서 O(1)? 
- 교체 후 해당 선박의 공격력은 pw가 된다.
- 교체 된 선박은 바로 ships_lists에 넣어줘야 다음에 바로 공격 명령이 들어와도 사용가능하다. 
    - 예전 Ships은 lazy check로 나중에 valid check를 할 수 있다. 
    - dict에 있는 ships안의 정보만 바꾸면, queue안에 같은 id이고 같은 pw를 가진 object가 두 개가 존재하게 된다. 두번 넣었으므로. 
    - 따라서, 같은 id인지 확인만 해주면 된다. 

4. 공격 명령 (400)
- 최대 5000번까지 주어짐. 이 부분이 시간 복잡도의 최대 bottle neck일 것 같음. 
- 사격 대기 (0) 상태인 선박 중 공격력이 가장 높은 선박 최대 5척에 일제 사격을 명령한다. 
    - 최대 5척이므로 사용가능한 선박의 수가 그것보다 작을 경우에는 더 적을 수도 있다. 
    - 재장전 중인 선박은 공격에 사용불가 
        - 해당 선박이 현재 시간이 공격 후 r시간이 지났으면 사격 대기로 바꿀 수 있음. 
        - 만약 선박의 재장전 시간이 1초이면 1초에 공격했으면 2초때 바로 공격 가능 (즉, 시간 차이 >= r 임.)
    - 우선순위 
        - 공격력이 높다. 
        - 공격력이 같다면, 선박 번호 id가 작은 선박 
        - 총 피해가 최대가 되도록 선박을 고른다. (피해 = 사격에 참여한 선박들의 공격력 합)
    - 맨 앞의 5개를 꺼낼 때 sort()이면 O(1)이지만, heapq를 쓰면 O(5logN)시간이 걸림. 
        - sort(): Tlog(T): 앞에 5개의 원소에 접근 및 마지막에 다시 sort해줄 때 Tlog(T)시간이 걸림 
            - sort()로 해야할 것 같음. 
        - heapq: ~ 3 * 1e6 * 2 (제거하고 다시 사격대기로 전환해서 삽입시켜줘야함) 5T(logT)
- 사격에 참여한 선박들의 공격력 합만큼 대형 함선에 피해를 준다. 
    - Total_attacked: int = 대형 함선이 가진 피해
- 사격한 선반은 즉시 재장전에 들어가며, 사격 시점을 포함해 r시간이 경과하면 다시 '사격 대기(0)' 상태로 전환된다. 

- 공격 명령이 떨어지면, 아래의 3개 출력 
    - '해당 차례'의 총 피해량 
    - 사격에 참여한 선박 수(최대 5척)
    - 사격 우선순위에 따른 사격 선박들의 id 


Algorithm for 400 
1-1. 사용한 ship들에 대해 재정비 시간이 만료된 ship들에 대해서 queue에 다시 삽입한다. 
    - 이때 재정비된 Ships list에서 제거한다. 
    - status 0, used_time = -1 인 친구들 다시 삽입 
2-1. queue의 맨 앞에서부터 '최대' 5개를 뽑는다. 
    - 이번에 사용할 id와 같은 것이 있으면 pass 하면서 최대 5개를 뽑는다. 
        - 이때는 중복된 object이므로 버린다. 
    - 교체된 pw와 같은지 id_to_ships에서 정보를 확인하고 같지 않으면 pop -> object sharing이라서 같은지 아닌지 확인안해도됨. 
        - (어차피 status, pw, id순으로 되어 있기 때문에, 위의 스텝에서 정확히 재정비된 것들을 다시 넣었다면, 맨 앞의 5개가 맞음)
    -  다음 3개의 정보를 저장하고 출력한다. 
        - '해당 차례'의 총 피해량 
        - 사격에 참여한 선박 수(최대 5척)
        - 사격 우선순위에 따른 사격 선박들의 id 
3-1. 사용된 함선들은 used_ships에 저장한다. 
'''

import math 

# 공격 명령이 총 T번 (최대 5*1e4) 일때 우선순위 큐를 사용하면 cmd 4의 시간 복잡도  
T = 5*1e3
N = 30000
# print(T*5 * math.log(T)) # 3 * 1e6
# print(T * math.log(T))
# print(N * math.log(N)) # 3 * 1e5
# print(N*math.log(N) * T)
print(1e4* math.log(4e4)) # 1 * 1e5


# class Ship:
#     def __init__(self, id: int, p: int, r:int):
#         self.id = id 
#         self.p = p 
#         self.r = r 
#         self.status = 0 
#     def __repr__(self):
#         return f"Ship({self.id} with p {self.p})"

# my_ship1 = Ship(1, 2, 3)
# my_ship2 = Ship(2, 3, 4)

# ship_list = [my_ship1, my_ship2]
# ship_dict = {
#     my_ship1.id: my_ship1,
#     my_ship2.id: my_ship2
# }
# def change_ship(cur_ship, pw):
#     cur_ship.p = pw 

# print(f"Before {ship_list}")
# # change_ship(ship_list[1], 10)
# change_ship(ship_dict[1], 10)
# print(f"After {ship_list}")


import heapq 

class Ship:
    def __init__(self, id:int, p: int, r: int):
        self.id = id 
        self.p = p
        self.r = r 
        self.used_time = -1 # 사용되었던 시간 
        self.status = 0 # 대기 0, 재장전 1 

    def __lt__(self, other):
        if self.status != other.status: 
            return self.status < other.status # 대기하는 선박의 우선순위가 높음. 
        if self.p != other.p:
            return self.p > other.p 
        return self.id < other.id 
    
    def can_change_to_ready_status(self, cur_t: int):
        # 재정비 시간 r이 지나면 재정비 가능 
        return cur_t - self.used_time >= self.r 
    
    def reinit(self):
        self.status = 0
        self.used_time = -1 

    # def __repr__(self):
    #     return f"Ship(id: {self.id})(p: {self.p})(r: {self.r})"

# ship1 = Ship(1, 10, 3)
# ship2 = Ship(2, 1, 3)
# cur_t = 10 
# ships_pq = []
# used_ship_list = [ship1, ship2]

# used_ship_list_copy = used_ship_list[:]
# for used_ship in used_ship_list_copy:
#     if used_ship.can_change_to_ready_status(cur_t=cur_t):
#         used_ship.reinit()
#         heapq.heappush(ships_pq, used_ship)
#         # used_ship을 제거해야하는데 for loop돌고 있는 상황에서 제거하면 길이가 달라져서 안됨. 
#         # 따라서 copy 를 만들어서 제거해줌 
#         # NOTE: 원본에서 제거해도 used_ship_list_copy는 그대로 인지 확인 
#         used_ship_list.remove(used_ship)

ship1 = Ship(1, 10, 3)
ship2 = Ship(2, 11, 4)

my_dict: dict[int, Ship] = dict()
my_dict[1] = ship1 
my_dict[2] = ship2

pq = [ship2]


cur_ship = heapq.heappop(pq)
ship3 = Ship(3, 12, 5)
# cur_ship = ship3  # 이렇게 하면 변수가 가리키는 것만 바뀌어서 안됨. 
my_dict[cur_ship.id] = ship3 

print(my_dict)

my_list= []

my_list.append(my_dict[cur_ship.id])

print(my_list[0])
print(my_dict[cur_ship.id])


```

````
````{admonition} object sharing 
:class: dropdown 

파이썬에서는 같은 Object를 List나 dictionary에 저장해두고, 
dictionary 안의 object안의 variable을 변환해주면, 이를 가지고 있던 List 안의 같은 object도 변해있음을 확인할 수 있다. 
이는 파이썬이 객체를 sharing하는 기능을 가졌기 때문에 가능하다. 아래 코드로 확인할 수 있다. 

```{code-block} python 
class Ship:
    def __init__(self, id: int, p: int, r:int):
        self.id = id 
        self.p = p 
        self.r = r 
        self.status = 0 
    def __repr__(self):
        return f"Ship({self.id} with p {self.p})"

my_ship1 = Ship(1, 2, 3)
my_ship2 = Ship(2, 3, 4)

ship_list = [my_ship1, my_ship2]
ship_dict = {
    my_ship1.id: my_ship1,
    my_ship2.id: my_ship2
}
def change_ship(cur_ship, pw):
    cur_ship.p = pw 

print(f"Before {ship_list}")
# change_ship(ship_list[1], 10)
change_ship(ship_dict[1], 10)
print(f"After {ship_list}")
```
````

````{admonition} copy of a list 
:class: dropdown 

used_ship을 제거해야하는데 for loop돌고 있는 상황에서 제거하면 길이가 달라져서 안됨.  따라서 copy 를 만들어서 제거해줌 
즉, 원본에서 제거해도 used_ship_list_copy는 그대로 인지 확인 

```{code-block} python 
ship1 = Ship(1, 10, 3)
ship2 = Ship(2, 1, 3)
cur_t = 10 
ships_pq = []
used_ship_list = [ship1, ship2]

used_ship_list_copy = used_ship_list[:]
for used_ship in used_ship_list_copy:
    if used_ship.can_change_to_ready_status(cur_t=cur_t):
        used_ship.reinit()
        heapq.heappush(ships_pq, used_ship)
        # used_ship을 제거해야하는데 for loop돌고 있는 상황에서 제거하면 길이가 달라져서 안됨. 
        # 따라서 copy 를 만들어서 제거해줌 
        # NOTE: 원본에서 제거해도 used_ship_list_copy는 그대로 인지 확인 
        used_ship_list.remove(used_ship)
```
````

````{admonition} 여러 가지 데이터 구조안의 object 맞춰주기 
:class: dropdown 

이 문제의 경우 cmd 300에서 power를 바꿔주면, dictionary는 바로 업데이트할 수 있지만, ships_pq나 used_ship에서는 바로 바꾸기 힘들다. 

따라서 lazy evaluation을 진행하는데, 
문제는 heapq에 push할때 object sharing이 되어 있는 원소를 넣으면 둘은 완전히 동일한 값의 object이기 때문에 ships_pq 가 의도한대로 갱신되지 않는다. 

예를 들어, ship(2, 3, 2)가 pw=13으로 제일 커져서 맨 위로 올라가야한느데, 이미 pq에 있는 ship(2, 3, 2)가 ship(2, 13, 2)로 바뀌고 순위는 변동이 없기 때문에 새로이 ship(2, 13, 2)을 넣어도 원하는대로 순위 변동이 일어나지 않는다. 

따라서, 새로운 기존 Ship 정보를 바꾸지 않고, 새로운 object 생성 후 넣어준 후, dictionary값을 새로운 Ship으로 교체한다. 

그리고 난후, pq나 list에서 정보를 뽑을 때, lazy deletion 만약, 해당 Ship의 power가 바뀌었거나 status가 이미 변했으면, dictionary 값을 통해 삭제하거나 무시한다. 

아래는 dictionary값을 새로운 Ship으로 교체해야함을 알려준다. 기존 값을 바꾸면 두 가지 object가 다르므로 혼란이 온다. 

```{code-block} python 

import heapq 

class Ship:
    def __init__(self, id:int, p: int, r: int):
        self.id = id 
        self.p = p
        self.r = r 
        self.used_time = -1 # 사용되었던 시간 
        self.status = 0 # 대기 0, 재장전 1 

    def __lt__(self, other):
        if self.status != other.status: 
            return self.status < other.status # 대기하는 선박의 우선순위가 높음. 
        if self.p != other.p:
            return self.p > other.p 
        return self.id < other.id 
    
    def can_change_to_ready_status(self, cur_t: int):
        # 재정비 시간 r이 지나면 재정비 가능 
        return cur_t - self.used_time >= self.r 
    
    def reinit(self):
        self.status = 0
        self.used_time = -1 

ship1 = Ship(1, 10, 3)
ship2 = Ship(2, 11, 4)

my_dict: dict[int, Ship] = dict()
my_dict[1] = ship1 
my_dict[2] = ship2

pq = [ship2]


cur_ship = heapq.heappop(pq)
ship3 = Ship(3, 12, 5)
# cur_ship = ship3  # 이렇게 하면 변수가 가리키는 것만 바뀌어서 안됨. 
my_dict[cur_ship.id] = ship3 

print(my_dict)

my_list= []

my_list.append(my_dict[cur_ship.id])

print(my_list[0])
print(my_dict[cur_ship.id])


```
````

````{admonition} solution
:class: dropdown 

```{code-block} python
import sys 
import heapq 

# sys.stdin = open('Input.txt')
input = sys.stdin.readline 

class Ship:
    def __init__(self, id:int, p: int, r: int):
        self.id = id 
        self.p = p
        self.r = r 
        self.used_time = -1 # 사용되었던 시간 
        self.status = 0 # 대기 0, 재장전 1 

    def set_used_time(self, used_time: int):
        self.used_time = used_time 
    def set_status(self, status: int):
        self.status = status 

    def __lt__(self, other):
        if self.status != other.status: 
            return self.status < other.status # 대기하는 선박의 우선순위가 높음. 
        if self.p != other.p:
            return self.p > other.p 
        return self.id < other.id 
    
    def can_change_to_ready_status(self, cur_t: int):
        # 재정비 시간 r이 지나면 재정비 가능 
        assert self.status == 1 # 이미 사용된 상태여야한다. 
        return cur_t - self.used_time >= self.r 
    
    def reinit(self):
        self.status = 0
        self.used_time = -1 

def process_100(N: int, ships_info: list[int]):
    global ships_pq, ship_id_to_dict
    for idx in range(2, 3*N, 3):
        # print(idx)
        cur_id, cur_pw, cur_r = ships_info[idx-2], ships_info[idx-1], ships_info[idx]
        cur_ship = Ship(id=cur_id, p=cur_pw, r=cur_r)
        heapq.heappush(ships_pq, cur_ship) # object sharing 
        ship_id_to_dict[cur_id] = cur_ship 

def process_200(id: int, p:int, r:int):
    global ship_id_to_dict, ships_pq 
    new_ship = Ship(id=id, p=p, r=r)
    # object sharing 
    ship_id_to_dict[id] = new_ship # id는 중복되지 않음 
    heapq.heappush(ships_pq, new_ship)

def process_300(id: int, new_pw: int, cur_t: int):
    global ship_id_to_dict, ships_pq

    # ship_id_to_dict[id].p = new_pw # object sharing이 되어 있어서 pq안에도 바뀜 
    # 300 명령 다음에 바로 400하는 경우, 재정렬이 안됨. 즉, 위에서 교체만 한다고해서, 재정렬이 안되기 때문에, 가장 우선순위가 높은 것이 앞에오도록 삽입해줘야함. 
    # NOTE: 원소를 바로 넣는 것도 object sharing 임. 

    # 교체했으면 dictionary에 있는 것과 현재 heapq로 넣는게 또 같아져버림. 
    # 우선순위가 제대로 되어있게 하기 위해 새로운 것을 넣어줌. 
    # NOTE: 이전에 있는 것이 잘못되어 있으면 버려야함. (lazy check)
    # 근데 이때 바뀔 ship이 사용상태가 0인지 1인지 모름 
    new_ship = Ship(ship_id_to_dict[id].id, new_pw, ship_id_to_dict[id].r)
    new_ship.set_status(ship_id_to_dict[id].status)
    new_ship.set_used_time(ship_id_to_dict[id].used_time)
    if new_ship.status == 1 and new_ship.can_change_to_ready_status(cur_t):
        new_ship.reinit()

    if new_ship.status == 0:
        heapq.heappush(ships_pq, new_ship) # 이것도 object sharing인가? YES 
    else: # new_ship.status == 1:
        used_ship_list.append(new_ship) # 추후에 lazy detection 필요 

    ship_id_to_dict[ship_id_to_dict[id].id] = new_ship

def process_400(cur_t):
    global used_ship_list, ships_pq, ship_id_to_dict

    # 이미 공격한 함선 중, 다시 재정비 할 수 있는 ship들을 ships_pq에 삽입 
    used_ship_list_copy = used_ship_list[:]
    for used_ship in used_ship_list_copy:
        # lazy deletion over cmd 300 
        if used_ship.status != ship_id_to_dict[used_ship.id].status:
            # NOTE: 원본에서 제거해도 used_ship_list_copy는 그대로!
            used_ship_list.remove(used_ship)
            continue
        # 이미 300에서 변한 경우에 ship_pq에 넣어줘서 괜찮음. 
        if used_ship.p != ship_id_to_dict[used_ship.id].p:
            used_ship_list.remove(used_ship)
            continue 

            # 

        if used_ship.can_change_to_ready_status(cur_t=cur_t):
            # if used_ship.id == 6:
            #     print('a')
            
            # NOTE: 만약, used_ship.init()을 먼저 하고 heappush하면, 기존에 heapq에 있던 것도 똑같이 바뀌어서 동일한 값을가진 객체가 들어가므로 새롭게 갱신이 안됨. 
            new_ship = Ship(used_ship.id, used_ship.p, used_ship.r)
            # 이렇게 new_ship을 우선순위 큐에 넣으면 dictionay가 가진 Object와 priority_queue가 가진 object가 달라지게됨. 
            heapq.heappush(ships_pq, new_ship) # 
            # used_ship.reinit() # dict에서 정보를 바꿔줌. 
            ship_id_to_dict[used_ship.id] = new_ship
            # used_ship을 제거해야하는데 for loop돌고 있는 상황에서 제거하면 길이가 달라져서 안됨. 
            # 따라서 copy 를 만들어서 제거해줌 
            # NOTE: 원본에서 제거해도 used_ship_list_copy는 그대로!
            used_ship_list.remove(used_ship)


    will_be_used_set = set() # O(1)으로 중복 찾기 위함 
    will_be_used_list = [] # 우선순위대로 프린트 하기 위함 
    will_not_be_used = [] # 조건이 안 된 것들을 저장하기 위함 
    total_pw = 0
    while ships_pq: # 5개 도달하기 전에 ships_pq가 비어있으면 종료 
        cur_ship = heapq.heappop(ships_pq)

        # 오래된 것은 버려야함. (lazy check)
        if ship_id_to_dict[cur_ship.id].p != cur_ship.p:
            continue 

        if cur_ship.id in will_be_used_set:
            # 중복된 object이므로 제거되어도 됨. 
            continue 
        
        # 최대 5개인데, 5개 중 맨 뒤에 2개가 이미 사용된 ships일수도 있잖아. 
        # 하지만 사용하면 바로 빼서 그럴 일은 없을 것 같음. 
        if cur_ship.status == 1:
            # 제거하진 않고, 나중에 다시 넣어줌. 
            will_not_be_used.append(cur_ship)

        
        # 공격에 들어갈 것. dict에 있는 정보도 바뀌게 됨. 
        # 이미 ships_pq에서는 사용되면 빠지게 됨. 즉, Ships_pq에는 대기 (status=0) 중인 선박만 존재 # NOTE: cmd 300에서 이미 사용되었던 ship이라도 pq에 들어갈 수 있음. 
        cur_ship.status = 1 
        cur_ship.used_time = cur_t
        used_ship_list.append(cur_ship)

        #
        total_pw += cur_ship.p 
        will_be_used_set.add(cur_ship.id)
        will_be_used_list.append(cur_ship.id)
        # heapq.heappop(ships_pq)

        # 최대 5개 까지 
        if len(will_be_used_set) >= 5:
            break 
        
    print(total_pw, len(will_be_used_set), end=' ')
    for id in will_be_used_list:
        print(id, end=' ')
    print()
    

    ### will_not_be_used를 다시 pq에 넣어줌 
    for ship in will_not_be_used:
        heapq.heappush(ships_pq, ship)

#### global scope 
if __name__ == "__main__":
    T = int(input())
    '''
    필요한 자료 구조
    '''
    ships_pq: list[Ship] = [] # priority queue 
    ship_id_to_dict: dict[int, Ship] = dict()
    used_ship_list: list[Ship] = []

    for t in range(T):
        cmd = list(map(int, input().split()))

        if cmd[0] == 100: # INIT 
            process_100(cmd[1], cmd[2:])

        elif cmd[0] == 200: # 지원 요청 
            process_200(cmd[1], cmd[2], cmd[3])

        elif cmd[0] == 300: # 함포 교체 
            id = cmd[1]
            new_pw = cmd[2]
            process_300(id, new_pw, t)
        else: # 400, 공격 명령 
            process_400(t)
```
````

## Dijkstra 

특정 시작점에서 다른 점으로의 최단 거리를 알고 싶은 경우, ***음수의 간선이 없는 경우*** dijkstra 알고리즘을 사용하여 계산할 수 있다. 

### 개구리의 여행 

1. 3D dijkstra algorithm을 사용할 수 있다. 즉, 공간적 위치 뿐만 아니라, (y, x, jump) 현재의 점프력에 따라서도 도착지점까지의 최단 거리 (시간)이 달라지기 때문이다. 따라서, Shortest_path dictionary와 priority queue에 넣는 정보 모두 3D 차원에서 고려, 확인해야한다. 
2. 시간 초과가 나는 경우, dijkstra algorithm에서 중간에, Destination에 도달했다면, 빨리 알고리즘을 종료시킴으로써, 해결할 수 있다. (다만 이경우에는 start_node가 동일한 경우 기존에 계산한 것에서 사용하지못하고, 다시 계산해야한다. )
   - 다익스트라는 ***우선순위 큐에서 pop되는 순간, 그 상태의 거리는 '그 상태로 가는 최단 거리'가 확정*** 이다. 
   - 나중에 다른 점프력 (d_y, d_x, j2)로 도달하는 경로들이 있을 수 있지만, 그 상태는 (d_y, d_x, j1)보다 더 작은 거리를 가지고 있어야하지만, pop()되어서 나온 것이 j1이면, 그 상태에서 최단 거리를 가지고 있기 때문에 고려하지 않아도 된다. 
3. 전체 상태의 개수는 격자 칸수와 점프력의 가능한 값의 곱에 비례하며, 다익스트라 알고리즘을 통해 처리하므로 시간복잡도는 O(N^2 J^2 log(N^2 J^2)) ~ O(N^2 J^2 log(N J)) 가 된다. 

````{admonition} coding and decoding for each state 
:class: dropdown 

공간상으로 3D Matrix을 만드는 것이 가장 쉬운 접근법이지만, (y, x, jump)에 대하여 unique한 index를 만드는 함수를 구현하여 코드를 작성할 수도 있다. 

```{code-block} python 
---
caption: 3D 공간 (행, 열, 점프력)을 하나의 stateID로 coding하는 함수. (row, col, jump)가 1-indexed이기 때문에, 0-indexed로 변환한 후 계산하고 있음에 주의하자.  
---
# 각 상태는 (행, 열, 점프력)으로 저장된다. 
# 상태를 하나의 정수 인덱스로 변환하기 위한 함수이다. 
# 상태 인덱스는 MAX_JUMP_POWER * (row-1) 
def getStateId(row: int, col:int, jump: int) -> int:
    global gridSize, MAX_JUMP_POWER
    # 1-indexed (row, col, jump)
    '''
    row에 gridSize 숫자를 곱하면, 그 숫자들이 gridSize만큼 벌어지고, 그것을 col-1의 크기만큼 채우면 Unique한 수를 만들 수 있다. 
    이는 Jump라는 3번째 숫자가 있을때도 동일하게 적용될 수 있다. 
    '''
    return MAX_JUMP_POWER*(gridSize*(row-1) + (col-1)) + (jump-1)
```

반면, 이렇게 unique index로 coding된 것들을 다시, (row, col, jump)로 계산할 수 있다. 

```{code-block} python

def decodeState(cur_state: int):
    tempState = cur_state 
    currentJumpPower = (tempState)%MAX_GRID_SIZE + 1 # 점프력 값 복원 
    tempState //= MAX_JUMP_POWER
    currentCol = (tempState % gridSize) + 1 # 열 복원 
    tempState //= gridSize
    currentRow = (tempState % gridSize) + 1 # 행 복원
     
    return (currentRow, currentCol, currentJumpPower)
```

````

````{admonition} Tips for dijkstra alogrithm 
:class: tip 
dijkstra priority queue에 (dis, (y, x)) 정보만 들어가면, 같은 위치에서 점프력이 다를때 중복되어 알고리즘이 정확히 움직이는 것을 파악하기 어렵지만, (dis, (y, x, jump))까지 들어가면, 겹치지 않고, 해당 State에 대해 최단 거리를 구할 수 있게 되므로, 굳이, options들을 구할때 점프후까지 고려할 필요가 없다. 

따라서, 위치 정보 이외에도 어떤 정보가 필요한지, 잘 고려하여 해당 정보도 포함하도록 넣어주어야한다. 
````

````{admonition} sol1: Time Limit 
:class: dropdown 

아래는 Time limit이 걸렸으나, 로직 자체는 맞는 것 같다. 시간초과가 나는 부분은 `cal_options()`함수를 호출할 때 부분으로, graph에 갈 공간이 많을 수록 할 수 있는 점프 및 다양한 경로가 존재하게 되어 이를 찾는데 시간초과가 걸리는 것 같다. 

현재 Time Limit이 나는 이유는, 다음에 갈 상태가 점프 후까지 계산을 해서 그런 것 같음. 
check_ways() 함수는 for loop을 진행하는 함수인데, 점프 옵션 외에 점프력 감소/증가할때도 따지게 되므로, 시간 초과되는 것 같음.
너무 멀리내다보지 말고, 현재 상황까지만 보도록 코드를 다시 짜보자. 

Tips:  dijkstra priority queue에 (dis, (y, x)) 정보만 들어가면, 같은 위치에서 점프력이 다를때 중복되어 알고리즘이 정확히 움직이는 것을 파악하기 어렵지만, (dis, (y, x, jump))까지 들어가면, 겹치지 않고, 해당 State에 대해 최단 거리를 구할 수 있게 되므로, 굳이, options들을 구할때 점프후까지 고려할 필요가 없다.



```{code-block} python 
import sys 
import heapq 

# sys.stdin = open('Input.txt')
input = sys.stdin.readline

class Node:
    def __init__(self, time: int, y: int, x:int, jump: int):
        self.time = time 
        self.y = y
        self.x = x 
        self.jump = jump
        
    def __repr__(self):
        return f"({self.y}, {self.x} with jump {self.jump})"
    
def modified_dijkstra(s_y:int, s_x:int, d_y:int, d_x: int):
    global shortest_path, options
    if (s_y, s_x) in shortest_path:
        dis = min(shortest_path[(s_y, s_x)][d_y][d_x])
        print(dis if dis != MAX else -1) 
        return 
    # 해당 시작 노드에서 계산한 shortest_path가 없는 경우 
    shortest_path[(s_y, s_x)] = [[[MAX] * 6 for _ in range(1+N)] for _ in range(1+N)]
    shortest_path[(s_y, s_x)][s_y][s_x][1] = 0 

    q = []
    heapq.heappush(q, (0, (s_y, s_x), 1)) # dis, cur_locs, jump
    # min_dis = MAX 
    visited = set() # options를 위한 방문 처리 셋 
    while q:
        cur_dis, cur_locs, cur_jump = heapq.heappop(q)
        cur_y = cur_locs[0]; cur_x = cur_locs[1]

        # NOTE: 같은 칸이라고 해도 Jump=1, jump5일때 그 이후에 갈 수 있는 다음 칸/비용이 달라지므로 점프력도 포함해야한다. 
        if cur_dis > shortest_path[(s_y, s_x)][cur_y][cur_x][cur_jump]:
            continue 

        # backtracking 
        # print(f'cur_locs: {cur_y}, {cur_x}: {options[cur_y][cur_x][cur_jump]}')
        
        # 현재 locs와 현재 점프력에서 nxt_node에는 (edge_weight, 연결된 Node위치, 연결된 Node위치까지 걸리는 점프력) 저장 
        
        if (cur_y, cur_x, cur_jump) not in visited:
            options[cur_y][cur_x][cur_jump] = cal_options(cur_y, cur_x, cur_jump)
            visited.add((cur_y, cur_x, cur_jump))

        for nxt_node in options[cur_y][cur_x][cur_jump]:
            nxt_jump = nxt_node.jump # 다음 상태에서의 점프력
            nxt_time = cur_dis + nxt_node.time 
            
            if nxt_time < shortest_path[(s_y, s_x)][nxt_node.y][nxt_node.x][nxt_jump]:
                shortest_path[(s_y, s_x)][nxt_node.y][nxt_node.x][nxt_jump] = nxt_time 
                heapq.heappush(q, (nxt_time, (nxt_node.y, nxt_node.x), nxt_jump))

    # 결과 출력 
    min_dis = min(shortest_path[(s_y, s_x)][d_y][d_x])
    print(min_dis if min_dis != MAX else -1) 

def in_range(y, x):
    global N
    return 1 <= y <= N and 1 <= x <= N

def check_ways(cur_y: int, cur_x: int, dy: int, dx: int) -> bool:
    global graph 

    # condition1, 2 에서 도착 위치의 돌 정보를 확인하므로, 
    # 시작~끝의 '경로'에만 천적이 있는지 없는지 확인하면 됨. (도착위치는 exclusive)

    # dx나 dy가 0이면 range가 안돌아감. 
    if dy == 0:
        # dx방향으로만 검사 
        dir = -1 if dx < 0 else 1 
        for x in range(cur_x, cur_x+dx, dir):
            if '#' == graph[cur_y][x]:
                return False 
    elif dx == 0:
        dir = -1 if dy < 0 else 1 
        for y in range(cur_y, cur_y+dy, dir):
            if '#' == graph[y][cur_x]:
                return False 
    return True 

def make_jump(weight: int, cur_y: int, cur_x: int, cur_jump: int):
    global graph 

    DY = [cur_jump, -cur_jump, 0, 0]
    DX = [0, 0, cur_jump, -cur_jump]
    
    cur_options = []
    for dy, dx in zip(DY, DX):
        nxt_y = cur_y + dy 
        nxt_x = cur_x + dx 
        
        if in_range(nxt_y, nxt_x):
            condition1 = graph[nxt_y][nxt_x] == '.' # 도착위치에 돌이 있음
            condition2 = graph[nxt_y][nxt_x] != 'S' # 도착위치가 미끄러운 돌이 아님 
            condition3 = graph[nxt_y][nxt_x] != '#' # 도착위치에 천적이 거주 
            condition4 = check_ways(cur_y, cur_x, dy, dx) # 현재위치에서 경로까지 천적이 살지 않는지 
            flag = condition1 and condition2 and condition3 and condition4
            
            if flag:
                # NOTE: edge의 정해진 weight에 대한, Node생성, 현재 점프력도 저장 
                cur_options.append(Node(weight, nxt_y, nxt_x, cur_jump))

    return cur_options

def cal_options(cur_y: int, cur_x: int, cur_jump: int) -> list[Node]:
    # 현재 위치와 점프력으로 '다음에' 갈 수 있는 (weight, nxt_y, nxt_x)의 정보 수집
    can_reach = []  
    # 1) 바로 점프  = 1
    can_reach += make_jump(weight=1, 
                           cur_y=cur_y , cur_x=cur_x, cur_jump=cur_jump)
    
    # 2) 점프력 증가 후 점프 = k^2 + 1 
    # NOTE: 점프력을 1올릴 수 있다고 했는데, 이는 만약 1을 올려도 없으면, 제자리에서 또 점프력을 올릴 수 있음 
    # if 1<= cur_jump <= 4:
    #     elevated_jump = cur_jump + 1 
    #     can_reach += make_jump(weight=1+elevated_jump*elevated_jump, 
    #                            cur_y= cur_y, cur_x=cur_x, cur_jump=elevated_jump)
    if 1<=cur_jump <= 4:
        weight = 0
        for elevated_jump in range(cur_jump+1, 6):
            # 누적합 
            weight += (elevated_jump*elevated_jump)
            can_reach += make_jump(weight=1+weight,
                                   cur_y = cur_y, cur_x = cur_x, cur_jump=elevated_jump)

    # 3) 점프력 감소 후 점프 = 1 + 1 
    for reduced_jump in range(1, cur_jump):
        can_reach += make_jump(weight=1+1, cur_y=cur_y , 
                               cur_x=cur_x, cur_jump=reduced_jump)
    return can_reach 



N = int(input())
MAX = int(1e9)
#### 필요한 자료구조 
graph = [[0]*(1+N)]
for idx in range(1, N+1):
    graph.append([0])
    graph[idx] = graph[idx] + list(input())

# print(graph)
# print(len(graph), len(graph[0]))

# 현재 위치 (tuple)에서 시작할때 각 도착지에 대해서 걸리는 최단 시간에 대한 정보 저장 
# 3D dijkstra, [y][x][jump]
shortest_path: dict[tuple, list[list[list[int]]]] = dict()
# 현재 위치와 점프력으로 '다음에' 갈 수 있는 (weight, nxt_y, nxt_x)의 정보 수집 
options: list[list[list["Node"]]] # [cur_y][cur_x][jump] -> [(edge weight(걸리는 시간), next_y, next_x, 도달할때 점프력), 저장]

options = [[[[] for _ in range(6)] for _ in range(1+N)] for _ in range(1+N)]

Q = int(input())

# NOTE: 이렇게 다 만들고 풀면, 시간 초과 
# # 모든 시작 위치에 대해서 
# for cur_y in range(1, N+1):
#     for cur_x in range(1, N+1):
#         for cur_jump in range(1, 6): # jump는 1에서 5까지
#             # 다시 돌아갈 수도 있는거잖아...아닌가?
#             # if cur_y == 1 and cur_x == 1 and cur_jump != 1:
#             #     # 최초 위치에서는 cur_jump이 1밖에 없음. 
#             #     continue 
#             options[cur_y][cur_x][cur_jump] = cal_options(cur_y, cur_x, cur_jump) # options 미리 만들어놓기 

# print(options[6][2][1])

for _ in range(Q):
    r1, c1, r2, c2 = list(map(int, input().split()))
    modified_dijkstra(r1, c1, r2, c2)
```
````

````{admonition} Solution 
:class: dropdown 

```{code-block} python 
import sys 
import heapq 

# sys.stdin = open('Input.txt')
input = sys.stdin.readline

class Node:
    def __init__(self, time: int, y: int, x:int, jump: int):
        self.time = time 
        self.y = y
        self.x = x 
        self.jump = jump
        
    def __repr__(self):
        return f"({self.y}, {self.x} with jump {self.jump})"
    
def modified_dijkstra(s_y:int, s_x:int, d_y:int, d_x: int):
    global shortest_path, options
    # if (s_y, s_x) in shortest_path:
    #     dis = min(shortest_path[(s_y, s_x)][d_y][d_x])
    #     print(dis if dis != MAX else -1) 
    #     return 
    # 해당 시작 노드에서 계산한 shortest_path가 없는 경우 
    shortest_path[(s_y, s_x)] = [[[MAX] * 6 for _ in range(1+N)] for _ in range(1+N)]
    shortest_path[(s_y, s_x)][s_y][s_x][1] = 0 

    q = []
    heapq.heappush(q, (0, (s_y, s_x), 1)) # dis, cur_locs, jump
    # min_dis = MAX 
    # visited = set() # options를 위한 방문 처리 셋 
    while q:
        cur_dis, cur_locs, cur_jump = heapq.heappop(q)
        cur_y = cur_locs[0]; cur_x = cur_locs[1]

        # NOTE: 같은 칸이라고 해도 Jump=1, jump5일때 그 이후에 갈 수 있는 다음 칸/비용이 달라지므로 점프력도 포함해야한다. 
        if cur_dis > shortest_path[(s_y, s_x)][cur_y][cur_x][cur_jump]:
            continue 
        
        if cur_y == d_y and cur_x == d_x:
            shortest_path[(s_y, s_x)][d_y][d_x][cur_jump] = cur_dis
            break
        # backtracking 
        # print(f'cur_locs: {cur_y}, {cur_x}: {options[cur_y][cur_x][cur_jump]}')
        
        # 현재 locs와 현재 점프력에서 nxt_node에는 (edge_weight, 연결된 Node위치, 연결된 Node위치까지 걸리는 점프력) 저장 
        
        # if graph[cur_y][cur_x] == '.' and (cur_y, cur_x, cur_jump) not in visited:
        #     options[cur_y][cur_x][cur_jump] = cal_options(cur_y, cur_x, cur_jump)
        #     visited.add((cur_y, cur_x, cur_jump))

        for nxt_node in options[cur_y][cur_x][cur_jump]:
            nxt_jump = nxt_node.jump # 다음 상태에서의 점프력
            nxt_time = cur_dis + nxt_node.time 
            
            if nxt_time < shortest_path[(s_y, s_x)][nxt_node.y][nxt_node.x][nxt_jump]:
                shortest_path[(s_y, s_x)][nxt_node.y][nxt_node.x][nxt_jump] = nxt_time 
                heapq.heappush(q, (nxt_time, (nxt_node.y, nxt_node.x), nxt_jump))

    # 결과 출력 
    min_dis = min(shortest_path[(s_y, s_x)][d_y][d_x])
    print(min_dis if min_dis != MAX else -1) 

def in_range(y, x):
    global N
    return 1 <= y <= N and 1 <= x <= N

def check_ways(cur_y: int, cur_x: int, dy: int, dx: int) -> bool:
    global graph 

    # condition1, 2 에서 도착 위치의 돌 정보를 확인하므로, 
    # 시작~끝의 '경로'에만 천적이 있는지 없는지 확인하면 됨. (도착위치는 exclusive)

    # dx나 dy가 0이면 range가 안돌아감. 
    if dy == 0:
        # dx방향으로만 검사 
        dir = -1 if dx < 0 else 1 
        for x in range(cur_x, cur_x+dx, dir):
            if '#' == graph[cur_y][x]:
                return False 
    elif dx == 0:
        dir = -1 if dy < 0 else 1 
        for y in range(cur_y, cur_y+dy, dir):
            if '#' == graph[y][cur_x]:
                return False 
    return True 

def make_jump(weight: int, cur_y: int, cur_x: int, cur_jump: int):
    global graph 

    DY = [cur_jump, -cur_jump, 0, 0]
    DX = [0, 0, cur_jump, -cur_jump]
    
    cur_options = []
    for dy, dx in zip(DY, DX):
        nxt_y = cur_y + dy 
        nxt_x = cur_x + dx 
        
        if in_range(nxt_y, nxt_x):
            condition1 = graph[nxt_y][nxt_x] == '.' # 도착위치에 돌이 있음
            condition2 = graph[nxt_y][nxt_x] != 'S' # 도착위치가 미끄러운 돌이 아님 
            condition3 = graph[nxt_y][nxt_x] != '#' # 도착위치에 천적이 거주 
            condition4 = check_ways(cur_y, cur_x, dy, dx) # 현재위치에서 경로까지 천적이 살지 않는지 
            flag = condition1 and condition2 and condition3 and condition4
            
            if flag:
                # NOTE: edge의 정해진 weight에 대한, Node생성, 현재 점프력도 저장 
                cur_options.append(Node(weight, nxt_y, nxt_x, cur_jump))

    return cur_options

def cal_options(cur_y: int, cur_x: int, cur_jump: int) -> list[Node]:
    # 현재 위치와 점프력으로 '다음에' 갈 수 있는 (weight, nxt_y, nxt_x)의 정보 수집
    can_reach = []  
    # 1) 바로 점프  = 1
    can_reach += make_jump(weight=1, 
                           cur_y=cur_y , cur_x=cur_x, cur_jump=cur_jump)
    
    # 2) 점프력 증가 후 점프 = k^2 + 1 
    # NOTE: 점프력을 1올릴 수 있다고 했는데, 이는 만약 1을 올려도 없으면, 제자리에서 또 점프력을 올릴 수 있음 
    # if 1<= cur_jump <= 4:
    #     elevated_jump = cur_jump + 1 
    #     can_reach += make_jump(weight=1+elevated_jump*elevated_jump, 
    #                            cur_y= cur_y, cur_x=cur_x, cur_jump=elevated_jump)
    if 1<=cur_jump <= 4:
        weight = 0
        for elevated_jump in range(cur_jump+1, 6):
            # 누적합 
            weight += (elevated_jump*elevated_jump)
            # can_reach += make_jump(weight=1+weight,
            #                        cur_y = cur_y, cur_x = cur_x, cur_jump=elevated_jump)
            # 점프력을 올리면, 점프력 상승만 하고, Weight 증가 but 그 자리에 가만히 있게 됨. 
            can_reach.append(Node(time=weight,
                                  y=cur_y, x=cur_x, jump=elevated_jump))

    # 3) 점프력 감소 후 점프 = 1 + 1 
    for reduced_jump in range(1, cur_jump):
        # can_reach += make_jump(weight=1+1, cur_y=cur_y , 
        #                        cur_x=cur_x, cur_jump=reduced_jump)
        can_reach.append(Node(time=1,
                              y=cur_y, x=cur_x, jump=reduced_jump))
    return can_reach 



N = int(input())
MAX = int(1e9)
#### 필요한 자료구조 
graph = [[0]*(1+N)]
for idx in range(1, N+1):
    graph.append([0])
    graph[idx] = graph[idx] + list(input())

# print(graph)
# print(len(graph), len(graph[0]))

# 현재 위치 (tuple)에서 시작할때 각 도착지에 대해서 걸리는 최단 시간에 대한 정보 저장 
# 3D dijkstra, [y][x][jump]
shortest_path: dict[tuple, list[list[list[int]]]] = dict()
# 현재 위치와 점프력으로 '다음에' 갈 수 있는 (weight, nxt_y, nxt_x)의 정보 수집 
options: list[list[list["Node"]]] # [cur_y][cur_x][jump] -> [(edge weight(걸리는 시간), next_y, next_x, 도달할때 점프력), 저장]

options = [[[[] for _ in range(6)] for _ in range(1+N)] for _ in range(1+N)]

Q = int(input())

# NOTE: 이렇게 다 만들고 풀면, 시간 초과 
# # 모든 시작 위치에 대해서 
for cur_y in range(1, N+1):
    for cur_x in range(1, N+1):
        for cur_jump in range(1, 6): # jump는 1에서 5까지
            # 다시 돌아갈 수도 있는거잖아...아닌가?
            # if cur_y == 1 and cur_x == 1 and cur_jump != 1:
            #     # 최초 위치에서는 cur_jump이 1밖에 없음. 
            #     continue 
            if graph[cur_y][cur_x] == '.':
                options[cur_y][cur_x][cur_jump] = cal_options(cur_y, cur_x, cur_jump) # options 미리 만들어놓기 

# print(options[6][2][1])

for _ in range(Q):
    r1, c1, r2, c2 = list(map(int, input().split()))
    modified_dijkstra(r1, c1, r2, c2)
```
````


### Reachable Nodes In Subdivided Graph 

````{admonition} Solution
:class: dropdown 

Time: 111ms  <br>
Memory: 25 MB <br>

used: dict[(node1, node2), int]: for each edge, `used` dictionary stores the number of possible new nodes we can walk through within the `maxMoves`. maxMoves - shortest_dis to the cur node can be negative. For example, maxMoves=6 and the shortest path to the node 3 can be 9. and the `used` will store -3. (since -3 will be always smaller than the number of new nodes) <- This is why when we get out values in `used` by using used.get() function, we need to do used.get((u, v), 0). 

In the end, we calculate `ans += min(w, used.get((u, v), 0) + used.get((v, u), 0))` for each edge. 
Since the graph is undirected graph, we can walk from u to v and, also, from v to u. 

Since the addition of two values should not be greater than w (the number of new nodes), we add `min` value to the final answer. 

```{code-block} python 

import collections 
import heapq 

class Solution(object):
    def reachableNodes(self, edges, M, N):
        graph = collections.defaultdict(dict)
        for u, v, w in edges:
            graph[u][v] = graph[v][u] = w # how many new nodes there are on this edge 

        pq = [(0, 0)]
        dist = {0: 0}
        used = {}
        ans = 0

        while pq:
            d, node = heapq.heappop(pq)
            if d > dist[node]: continue
            # Each node is only visited once.  We've reached
            # a node in our original graph.
            ans += 1

            for nei, weight in graph[node].items():
                # M - d is how much further we can walk from this node;
                # weight is how many new nodes there are on this edge.
                # v is the maximum utilization of this edge.
                v = min(weight, M - d)
                used[node, nei] = v # start_node = node, end_node = nei

                # d2 is the total distance to reach 'nei' (neighbor) node
                # in the original graph.
                d2 = d + weight + 1 # (weight+1 = the number of edges)
                if d2 < dist.get(nei, M+1): # dict.get(key, value if there is no key)
                    heapq.heappush(pq, (d2, nei))
                    dist[nei] = d2

        # At the end, each edge (u, v, w) can be used with a maximum
        # of w new nodes: a max of used[u, v] nodes from one side,
        # and used[v, u] nodes from the other.
        for u, v, w in edges:
            ans += min(w, used.get((u, v), 0) + used.get((v, u), 0))

        return ans
```
````


````{admonition} Solution2 
:class: dropdown 

Time: 149ms 
Memory: 26MB 

```{code-block} python
from typing import List 
import heapq
from collections import defaultdict 

def modified_dijkstra(cur_node:int, maxMoves:int):
    global graph, shortest_path, used
    cnt = 0

    possible_reachable_nodes_num = maxMoves - (shortest_path[cur_node])
    # NOTE: possible reachable nodes num이 음수가 되면, 위에 Temp_num이 음수가 되어 
    # 로직이 틀려지므로 0으로 둔다. 
    # possible reachable nodes num이 음수 = 0보다 멀리가면 안되고, 0과 가까운쪽의 노드로 가야함. 
    # 근데 그쪽으로 가면, 어차피 이전에 Dijkstra에서 그쪽 노드에서 이미 계산했을 것이기 때문에, 그냥 지나치면 된다. 
    possible_reachable_nodes_num = 0 if possible_reachable_nodes_num < 0 else possible_reachable_nodes_num
    
    for nxt_node, num_1 in graph[cur_node].items():
        if cur_node < nxt_node:
            node1= cur_node; node2=nxt_node 
            
        else:
            node1 = nxt_node; node2=cur_node 
        
        exisiting_node_nums = used[(node1, node2)]
        if exisiting_node_nums == 0:
            continue 

        reachable_nodes_num = min(exisiting_node_nums, possible_reachable_nodes_num)
        temp_num = min(used[(node1,node2)], reachable_nodes_num)
        used[(node1, node2)] -= temp_num
        cnt += temp_num
    return cnt 

MAX = int(1e9)
class Solution:
    def reachableNodes(self, edges: List[List[int]], maxMoves: int, n: int) -> int:
        global graph, shortest_path, used
        # Step 1: graph Initialization with new edge weight 
        # NOTE: graph와 dijkstra 의 결과인 SHORTESt path모두 dictionary로 저장해, 
        # memory efficient 하게 만든다. 
        graph = defaultdict(dict)
        total = 0
        used = dict()

        for edge in edges:
            # undirected graph 
            graph[edge[0]][edge[1]] = graph[edge[1]][edge[0]] = (edge[2] + 1)
            used[(edge[0], edge[1])] = edge[2]
        
        ## Step 2: dijkstra Algorithm 
        dijkstra_pq = [(0, 0)] # start_dis, start_node 
        # modified_dijkstra_pq = []
        # n수가 많아지면, [MAX]*n은 좋지 않음. dictionary로 만듦. 
        shortest_path = dict()
        shortest_path[0] = 0 # start node 
         
        while dijkstra_pq:
            cur_dis, cur_node = heapq.heappop(dijkstra_pq)

            if cur_dis > shortest_path[cur_node]:
                continue 

            # each node is only visited once. we've reached a node in our original graph 
            if shortest_path[cur_node] <= maxMoves:
                total += 1 

            # 최단 거리가 계산된 노드부터, 이어져 있는 edge들에 대하여 New graph에 있는
            # reachable nodes들의 개수를 더해준다. 
            total += modified_dijkstra(cur_node, maxMoves)

            for nxt_node, nxt_weight in graph[cur_node].items():
                # nxt_node=nxt_state.nxt_node; nxt_weight = nxt_state.weight
                nxt_dis = cur_dis + nxt_weight 
                
                # MAX로 기본 값 세팅안되어있어서, 없으면 MAX값이라 됨. 
                if nxt_node not in shortest_path or nxt_dis < shortest_path[nxt_node]:
                    shortest_path[nxt_node] = nxt_dis 
                    heapq.heappush(dijkstra_pq, (nxt_dis, nxt_node))

        return total 
    

sol = Solution()
# edges = [[0,1,10],[0,2,1],[1,2,2]]; maxMoves = 6; n = 3 # 13 
# edges = [[0,1,4],[1,2,6],[0,2,8],[1,3,1]]; maxMoves = 10; n = 4 # 23 
# edges = [[1,2,4],[1,4,5],[1,3,1],[2,3,4],[3,4,5]]; maxMoves = 17; n = 5 # 1 
edges = [[1,2,5],[0,3,3],[1,3,2],[2,3,4],[0,4,1]]; maxMoves=7; n=5 # 13
print(sol.reachableNodes(edges, maxMoves, n))

```
````

### Second Minimum  Time to Reach Destination 

````{admonition} Solution Time Limit 
:class: dropdown 

```{code-block} python 
from typing import List 
import heapq 
from collections import defaultdict 
import math 

MAX = int(1e9)

'''Time complexity
원래의 Dijkstra 는 O(E+V logV) 이지만, 
Second shortest path를 찾는 경우에는, 달라짐. 

'''
def dijkstra(start: int, time:int, n:int, change:int):
    MAX = int(1e9)
    q = [(0, start)]
    shortest_path: dict = defaultdict(list)
    shortest_path[start] = [0]

    while q:
        cur_time, cur_node = heapq.heappop(q) # cur_idx: 횟수, 시간 계산을 위해 사용됨.

        # NOTE: Dijkstra에서 visited check하는 방식 
        # second shortest path를 찾기 전까지 버리는 element가 없음. 
        # 노드 n에 도달하기 전에, 다른 노드들에 도착할 때도 기다리고 다른 곳으로 가야할 때가 있음. 
        # N이 아닌 다른 노드들에 대해서는 3, 4 번째 계속 구하다가, 노드가 n인 경우에만 2번째까지 구하면 됨. 
        temp_shortest_path = shortest_path.get(cur_node, [])

        # NOTE: dijkstra 종료 조건: n node에서 마무리 
        # if cur_node == n:
        #     print(cur_node)
        # 해당 최종 노드로 들어오는 값이 동일한 경우에는, 맨 마지막을 반환해야함. "Strictly larger than the minimum value"
        if cur_node == n and len(set(temp_shortest_path)) == 2:
            return temp_shortest_path[-1] # second shortest path to node n 
        
        # cur_signal 계산 
        cur_idx, cur_signal = calculate_signal(cur_time, change)
        '''
        조건 (갈 수 있는 옵션 중 제한 조건):
        - 현재 시그널이 초록색: 바로 움직여야함. 어디로든 움직일 수 있음. (enter하는 것은 언제나 가능)
        - 현재 시그널이 빨강색: 떠날 수 없음. (Signal이 초록색인 경우에만 vertex를 움직일 수 있음.)
            -> 이 경우 기다려야하는데, 다음 시그널이 초록으로 바뀌는 시간까지만 기다리면 됨. 
        '''
        # 시그널이 빨간색인 경우 떠날 수 없음. 이 경우 다음 시그널인 초록으로 바뀌는 시간까지 버티면 됨.
        # 기다리는 시간은 shortest_path를 Update해주지 않음. 다음 노드로 가는 경우에만 업데이트 함.
        if cur_signal == 1:
            heapq.heappush(q, ((cur_idx+1) * change, cur_node))
        else: # 시그널이 초록색인 경우 어느곳으로라도 움직여야함.
            for nxt_node in graph[cur_node]: 
                nxt_time = cur_time + time 

                # if n == nxt_node: 
                shortest_path[nxt_node].append(nxt_time) # backtracking 
                # if n != nxt_node: # memory 아끼기 
                #     shortest_path[nxt_node][-1] = nxt_time 
                heapq.heappush(q, (nxt_time, nxt_node))
                
    

def calculate_signal(cur_time: int, change: int):
    idx = math.floor(cur_time / change)
    return (idx, 0 if idx % 2 == 0 else 1 )


class Solution:
    def secondMinimum(self, n: int, edges: List[List[int]], time: int, change: int) -> int:
        global graph 
        # create graph 
        graph = [[] for _ in range(n+1)]
        for edge in edges:
            graph[edge[0]].append(edge[1])
            graph[edge[1]].append(edge[0])

        return dijkstra(1, time, n, change)

        
sol = Solution()
n=5; edges=[[1,2],[1,3],[1,4],[3,4],[4,5]]; time=3; change=5 # 13 
# n=2; edges=[[1,2]]; time=3; change=2 # 11 
# n=7; edges=[[1,2],[1,3],[2,5],[2,6],[6,5],[5,7],[3,4],[4,7]]; time=4; change=7 # 22
print(sol.secondMinimum(n, edges, time, change))

# MAX_N = 1e4 
# MAX_edges = 2*1e4 
# print((MAX_N + MAX_edges) * math.log(MAX_N)) # ~ 3 * 10^5 
```
````

````{admonition} Solution
:class: dropdown 

When a signal is red, since we don't move to the next node, we don't push (time_taken, nxt_node) in the queue. We directly add them when we make a move to the nextnode. 

```{code-block} python 
from typing import List 
import heapq 
from collections import defaultdict 
import math 

MAX = int(1e9)


def dijkstra(start: int, time:int, n:int, change:int):
    MAX = int(1e9)
    q = [(0, start)]
    shortest_path = [[MAX]*2 for _ in range(n+1)]
    shortest_path[start][0] = 0
    freq = [0] * (n+1)

    while q:
        cur_time, cur_node = heapq.heappop(q) # cur_idx: 횟수, 시간 계산을 위해 사용됨.
        freq[cur_node] += 1 
        # NOTE: Dijkstra에서 visited check하는 방식 
        # second shortest path를 찾기 전까지 버리는 element가 없음. 
        # 노드 n에 도달하기 전에, 다른 노드들에 도착할 때도 기다리고 다른 곳으로 가야할 때가 있음. 
        # N이 아닌 다른 노드들에 대해서는 3, 4 번째 계속 구하다가, 노드가 n인 경우에만 2번째까지 구하면 됨. 

        # NOTE: dijkstra 종료 조건: n node에서 마무리 
        if cur_node == n and freq[cur_node] == 2:
            return shortest_path[n][1] # second shortest path to node n 
        
        # cur_signal 계산 
        cur_idx, cur_signal = calculate_signal(cur_time, change)
        '''
        조건 (갈 수 있는 옵션 중 제한 조건):
        - 현재 시그널이 초록색: 바로 움직여야함. 어디로든 움직일 수 있음. (enter하는 것은 언제나 가능)
        - 현재 시그널이 빨강색: 떠날 수 없음. (Signal이 초록색인 경우에만 vertex를 움직일 수 있음.)
            -> 이 경우 기다려야하는데, 다음 시그널이 초록으로 바뀌는 시간까지만 기다리면 됨. 
        '''
        # 시그널이 빨간색인 경우 떠날 수 없음. 이 경우 다음 시그널인 초록으로 바뀌는 시간까지 버티면 됨.
        # 기다리는 시간은 shortest_path를 Update해주지 않음. 다음 노드로 가는 경우에만 업데이트 함.
        if cur_signal == 1:
            # heapq.heappush(q, ((cur_idx+1) * change, cur_node))
            # (cur_idx+1) * change= green으로 바꾸는 시간 
            # + time = 다음 노드로 넘어가는 시간 
            nxt_time = (cur_idx+1) * change + time 
        else: # 시그널이 초록색인 경우 어느곳으로라도 움직여야함.
            nxt_time = cur_time + time 

        for nxt_node in graph[cur_node]:  
            # Ignore nodes that have already popped out twice, we are not interested in
            # visiting them again.
            if freq[nxt_node] == 2:
                continue 
            # 해당 최종 노드로 들어오는 값이 동일한 경우에는, 맨 마지막을 반환해야함. 
            # 즉, "Strictly larger than the minimum value"
            # 따라서, 동일 값이 있다면 무시하고 넘어가야함. 
            if shortest_path[nxt_node][0] > nxt_time:
                shortest_path[nxt_node][1] =  shortest_path[nxt_node][0]# backtracking 
                shortest_path[nxt_node][0] = nxt_time 
                heapq.heappush(q, (nxt_time, nxt_node))
            elif shortest_path[nxt_node][1] > nxt_time and shortest_path[nxt_node][0] != nxt_time:
                shortest_path[nxt_node][1] = nxt_time 
                heapq.heappush(q, (nxt_time, nxt_node))

def calculate_signal(cur_time: int, change: int):
    '''
    (change* idx <= time < change*(idx+1)) 안에서 그린/레드 시그널이 바뀜. 
    idx % 2 == 0일때는 그린, idx % 2 != 0 일때 레드 
    따라서, 
    idx <= time/change < idx + 1 
    이므로, 
    idx = floor(time/change)로 표현가능. 
    '''
    # idx = math.floor(cur_time / change)
    idx = (cur_time // change) # 위와 동일 // == math.floor 
    return (idx, 0 if idx % 2 == 0 else 1 )


class Solution:
    def secondMinimum(self, n: int, edges: List[List[int]], time: int, change: int) -> int:
        global graph 
        # create graph 
        graph = [[] for _ in range(n+1)]
        for edge in edges:
            graph[edge[0]].append(edge[1])
            graph[edge[1]].append(edge[0])

        return dijkstra(1, time, n, change)

        
sol = Solution()
n=5; edges=[[1,2],[1,3],[1,4],[3,4],[4,5]]; time=3; change=5 # 13 
# n=2; edges=[[1,2]]; time=3; change=2 # 11 
# n=7; edges=[[1,2],[1,3],[2,5],[2,6],[6,5],[5,7],[3,4],[4,7]]; time=4; change=7 # 22
n=12; edges=[[1,2],[1,3],[3,4],[2,5],[4,6],[2,7],[1,8],[5,9],[3,10],[8,11],[6,12]]; time=60; change=600 # 22
print(sol.secondMinimum(n, edges, time, change))

# MAX_N = 1e4 
# MAX_edges = 2*1e4 
# print((MAX_N + MAX_edges) * math.log(MAX_N)) # ~ 3 * 10^5 
```
````

### Minimum Weighted Subgraph With the Required Paths 

The idea is the following: paths from `s1` to dest and from `s2` to `dest` can have common point `x`. Then we need to reach:

1. From s1 to x, for this we use Dijkstra
2. From s2 to x, same.
3. From x to dest, for this we use Dijkstra on the reversed graph.
4. Finally, we check all possible x.

Remark
- In python it was quite challenging to get AC, and I need to look for faster implementation of Dijkstra, however complexity is still the same, it depends on implementation details.

Complexity
- It is O(n*log E) for time and O(n) for space.

```{admonition} Solution 
:class: dropdown 

```{code-block} python 
from collections import defaultdict 
from heapq import heappop, heappush 

class Solution:
    def minimumWeight(self, n, edges, s1, s2, dest):
        G1 = defaultdict(list)
        G2 = defaultdict(list)
        for a, b, w in edges:
            G1[a].append((b, w))
            G2[b].append((a, w))

        def Dijkstra(graph, K):
            q, t = [(0, K)], {}
            while q:
                time, node = heappop(q)
                if node not in t:
                    t[node] = time
                    for v, w in graph[node]:
                        heappush(q, (time + w, v))
            return [t.get(i, float("inf")) for i in range(n)]
        
        arr1 = Dijkstra(G1, s1)
        arr2 = Dijkstra(G1, s2)
        arr3 = Dijkstra(G2, dest)
        
        ans = float("inf")
        for i in range(n):
            ans = min(ans, arr1[i] + arr2[i] + arr3[i])
        
        return ans if ans != float("inf") else -1
    
sol = Solution()
n = 6; edges=[[0,2,2],[0,5,6],[1,0,3],[1,4,5],[2,1,1],[2,3,3],[2,3,4],[3,4,2],[4,5,1]]; src1=0; src2=1; dest=5 # 9
# n = 3; edges=[[0,1,1],[2,1,1]]; src1=0; src2=1; dest=2 # -1 
# n = 8; edges=[[4,7,24],[1,3,30],[4,0,31],[1,2,31],[1,5,18],[1,6,19],[4,6,25],[5,6,32],[0,6,50]]; src1=4; src2=1; dest=6 # 44 
# n = 5; edges=[[0,2,1],[0,3,1],[2,4,1],[3,4,1],[1,2,1],[1,3,10]]; src1=0; src2=1; dest=4 # 3 
print(sol.minimumWeight(n, edges, src1, src2, dest))
```
