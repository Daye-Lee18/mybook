'''
색깔 트리 

Constraints: 
- Time: 10 sec 
- Memory: 200 MB 

필요한 자료 구조:
direct_parents = list()
colors = list()
max_depths = list()
dicret_childrens = list(list()) # 이거 보관하는게 쉽징않을수도 있엄. 
possible_dpeths = list()

- 동적으로 노드를 추가하고, 색깔을 변경할 수 있는 명령 존재 
- k-ary tree (binary트리라고 안되어 있음.)
- 처음에는 아무 노드도 존재하지 않음. 
- 노드 속성 
    - 고유 번호 id 
    - 부모 노드 번호 p_id : -1이면 새로운 트리의 루트 노드
    - 색깔 color: 1빨, 2주, 3노, 4초, 5파 
    - 최대 깊이 max_depth : 해당 노드를 루트로 하는 서브트리의 최대 깊이 
        - 자기 자신 노드의 깊이는 1
    - TODO 가능 깊이:
        - min(parent노드의 possible depth - 1, 자신의 max_depth)
        - root node의 경우 = max_depth 

cmd (1): 트리에 노드 추가
- 기존 노드들의 Max_depth값으로 인해 새로운 노드가 추가됨으로써 "모순이 발생"하면, 
    > TODO: 모순 체크 
        - 현재 노드의 부모 possible depth == 1 인 경우 모순 
- 현재 노드는 추가하지 않는다. 
> 현재 노드에서 possible depth를 계산할 수 있을 것 같음. 
- 최대 2*1e4 번 진행 

cmd (2): 색깔 변경 
- 특정 노드 m_id 를 루트로 하는 서브트리의 "모든" 노드의 색깔을 지정된 색 color 로 변경 
- 최대 5*1e4 번 주어짐. 
> XXX : 서브 트리의 모든 노드를 효율적으로 색깔 변경하는 방법 고안 
    - 서브 트리는 해당 노드의 children 전체 집합을 의미 
    - children = [[] for _ in range(n)]으로 함
    - time complexity : ~O(N) * 5*1e4 

cmd (3): 색깔 조회 
- 특정 노드 m_id의 현재 색깔을 조회 
> O(1) 시간 복잡도로 조회하는 방법 있을까? 
> node_color[m_id] 

cmd (4): 점수 조회 
- 모든 노드의 가치를 계산한여, 가치 "제곱"의 합을 출력 
- 각 노드의 가치: 
    : 해당 노드를 루트로 하는 서브트리 내 서로 다른 색깔의 "수" 
- DFS로 leaf node부터 수를 계산해서 하면 될 것 같음 
    - time complexity O(N) ~ O(2*1e4)

'''

N = int(2*1e4 )
Q = 1e5 

print(N*Q) # 2*1e9 >= 1 sec 

import sys 

childrens = [[] for _ in range(N)] # 0.1 MB 
childrens[0] = [1, 2, 3]
childrens[1]= [4]*N
childrens[2].append([4]*N)
childrens[3].append([4]*N)
# print(childrens)
print(sys.getsizeof(childrens))
print(dir(childrens))
print(sys.getsizeof(childrens.__hash__))
print(sys.getsizeof(childrens.__sizeof__))

visited1 = set([1, 3])
v2 = set([2])

visited1.union(v2)
print(visited1)