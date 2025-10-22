# Midterm 

## short answer problems 

- 아래 문제에서 각 문장의 빈 곳에 들어갈 말을 적거나, 문제에서 요구하는 것에 대해 간단히 서술하시오. 

Q1. integer는 1개당 4B를 차지한다. 따라서, 256MB의 메모리 사용량 제한이 있는 경우에는 N=$10^8$ 정도여야한다. 또한, python의 경우 1 초에 $2* 10^7$ operations이 가능하다. 만약, N=10000이고 time limit이 1 sec인 문제를 풀 때, 알고리즘의 time complexity는 최대 O(NlogN)이 된다. 

Q2. 그래프는 크게 두 가지 방식으로 구현될 수 있다. 하나는, 행렬에 각 노드 i부터 j까지의 edge weight을 저장하는 인접 행렬 (adjacent matrix) 방식이고, 다른 하나는, 하나의 노드에 linked list처럼 노드를 i -> [j, k, ..]로 연결하는 인접 리스트(adjacent list)의 방식이 있다. 인접 리스트의 장점은 메모리 사용량이 적다는 것이고, 단점은 연결되어 있는 노드를 찾는데 시간이 오래걸린다는 것이다. 

Q3. Backtracking이란, 문제에서 주어진 조건을 따라 하나씩 가능한 답을 찾아가는 방법으로 brute-force와 다른 점은 특정 조건을 만족하지 않으면, 해당 candidate를 탐색하지 않고 다음 것을 탐색하여 pruning을 한다는 것이다. Backtracking에서 depth는 이제까지 거쳐온 elements의 수를 의미하며, branch는 우리가 선택할 수 있는 서로다른 옵션의 개수를 말한다. 예를 들어, 서로 다른 5명의 사람들 중 2명을 골라 배열하는 모든 경우의 수에서 최대 depth는 2가 되고, branch는 5명이된다. 

Q4. Dynamic programming이란, 큰 문제를 작은 문제로 쪼개어 푸는 것으로 작은 문제의 답이 큰 문제를 풀때도 유효한 정답이여야한다. 이때, memoization/tabulation을 통해, 불필요한 중복되는 계산을 방지한다. DP문제를 풀때 3가지 중요하게 결정해야할 것이 있는데, 첫번째는 state이며, 두번째는 what to store이며, 마지막으로 transition이다. state란 우리가 추가적으로 생성한 dp table의 각 index가 어떤 것을 의미하는지를 말하며, what to store는 각 state에 어떤 value를 저장하는 지 결정하는 것이다. 마지막으로 transition은 기존에 계산했던 작은 문제들의 답을 이용해 현재의 문제를 풀 수 있는 점화식 관계를 의미한다. 

Q5. 이번 학기에 배운 shortest path 문제를 푸는 방법은 Dijkstra 알고리즘와 floyd warshall algorithm이 있다. 첫번째 방법은 한 지점에서 다른 모든 지점까지의 path cost를 구하는 것으로, greedy algorithm의 일종으로, 각 스텝마다, 현재 노드까지 갈 수 있는 가장 짧은 path를 구하고 넘어가는 식이다. advanced python implementation을 위해서는 priority queue를 사용하고 min-heap을 통해 구현한다. python에서 제공하는 standard library는 `heapq`이다. 

Q6. Floyd warshall algorithm은 dynamic programming을 일종이다. 이 알고리즘은 모든 노드에서 자신을 제외한 다른 노드까지의 비용을 구하는 알고리즘이다. 구현을 할 때는, 3중 for loop으로 구현을 하는데, 먼저 dp table에서 직접적으로 연결되어있는 edge가 없으면 INF으로 그리고 edge가 있으면 그 edge의 비용으로 초기화시켜주며, 가장 outer loop인 k가 안쪽 for loop두개의 a, b의 intermediate node로 가정하였을 때, 그 비용이 더 작다면 update하는 방식으로 진행이된다. 

Q7. BFS vs. DFS <br>
BFS는 Breadth-First Search의 약자로, queue data structure를 사용하여 구현한다. 반면, DFS는 Depth-First Search의 약자로, 함수의 recursion을 사용해 구현한다. DFS에서 call되는 함수는 stack구조로 불리고 끝난다. BFS는 최단 경로를 구할 때 유리하고, DFS는 한쪽으로 끝까지 내려가 탐색하므로, 트리나 그래프의 깊은 구조 탐색에 유리하다. 

Q8. Complexity Ordering 

다음 중 time complexity가 빠른 순서대로 나열하시오. 
1. O($N^2$)
2. O(NlogN)
3. O($2^n$)
4. O(N)
5. O(logN)

답: 5 < 4 < 2 < 1 < 3


Q9. Heap<br>
Min-heap에서 최솟값이 루트에 위치한다.Python에서 heap을 구현할 때 사용하는 기본 라이브러리는 heapq 이다. heapq library에는 두 가지 기본 method를 제공하는데, 하나는 삽입시에 사용하는 heappush()이며, 다른 하나는 제일 작은 priority 값을 얻을 때 사용하는 heappop()이다. 원소 삽입 시에는 새로 넣을 아이템의 값과 부모 노드의 값을 비교하여, 부모가 더 크면 현재 position의 값을 부모의 값으로 update하는 sift-up연산을 사용해 heap property를 유지한다. 반면, 가장 큰 priority를 가진 원소를 제거할 때는, 원래 heap의 마지막 원소를 root자리에 넣고, 마지막 원소의 값과 현재 position의 두 children 노드 중 가장 작은 child와 비교하여, child가 더 작은 경우, 현재 position값을 child 값으로 업데이트하며 이 과정을 sift-down연산을 한다.

## problem solving problems 

- 이번 학기의 중간까지 배운 챕터에 대해서 easy level의 코딩 문제들을 나열하였다. 
- test case는 각 문제당 5개씩 주어지며, 실제 점수를 계산할때는 더 많은 test case를 이용해 점수를 계산할 것이므로, 문제를 풀 때, edge test cases들을 직접 생성하고, 맞는 지 확인한 후 제출하기 바란다. 

````{admonition} simulation
:class: dropdown 

[문제](https://leetcode.com/problems/robot-return-to-origin/?envType=problem-list-v2&envId=simulation)
````

````{admonition} BFS 
:class: dropdown 

[문제](https://leetcode.com/problems/invert-binary-tree/description/?envType=problem-list-v2&envId=breadth-first-search)
````

````{admonition} Backtracking
:class: dropdown 

[문제](https://leetcode.com/problems/binary-tree-paths/description/?envType=problem-list-v2&envId=backtracking)
````

````{admonition} DP
:class: dropdown 

[문제](https://leetcode.com/problems/min-cost-climbing-stairs/description/?envType=problem-list-v2&envId=dynamic-programming)
````

````{admonition} Shortest Path 
:class: dropdown

[문제](https://leetcode.com/problems/find-the-city-with-the-smallest-number-of-neighbors-at-a-threshold-distance/description/?envType=problem-list-v2&envId=shortest-path)
````