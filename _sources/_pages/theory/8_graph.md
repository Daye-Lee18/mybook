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

# Lecture 8-1. Graph Algorithm 

DFS/BFS와 최단 경로에서 다른 내용은 모두 그래프 알고리즘의 한 유형으로 볼 수 있다. 일단, 알고리즘 문제를 접했을 때 ***서로 다른 개체 (혹은 객체<font size='2'> Object </font>)가 연결되어 있다*** 는 것을 보면 가장 먼저 그래프 알고리즘을 떠올려야 한다. 

## 서로소 집합 

수학에서 ***서로소 집합*** <font size='2'>Disjoint Sets</font>이란 공통 원소가 없는 두 집합을 의미한다. 

![1](../../assets/img/graph/1.png)

### 서로소 집합 자료구조 

서로소 집합 자료구조란 ***서로소 부분 집합들로 나누어진 원소들의 데이터를 처리하기 위한 자료구조*** 이다. "union"과 "find" 2개의 연산으로 조작할 수 있다. 서로소 집합에서 "find"연산은 특정한 원소가 속한 집합이 어떤 집합인지 알려주는 연산, "union"연산은 합집합으로 2개의 원소가 포함된 집합을 하나의 집합으로 합치는 연산이다. 

서로소 집합 자료구조는 ***union-find 자료구조*** 라고 불리기도 한다. 두 집합이 서로소 관계인지를 확인할 수 있다는 말은 각 집합이 어떤 원소를 공통으로 가지고 있는지를 확인할 수 있다는 말과 같기 때문이다. 

**핵심 개념 요약**

```{list-table} disjoint set / union-find
:widths: 15 50 
:header-rows: 1 

* - 항목 
  - 설명 
* - 정의 
  - 서로 겹치지 않는 부분 집합들로 구성된 집합을 관리하는 자료구조 
* - 핵심 연산 
  - find(x): 원소 `x`가 속한 집합 (루트 노드) 찾기 <br> , union(a, b): 두 원소가 속한 집합을 하나로 합치기 
* - 주요 목적
  - 두 원소가 "같은 집합(=같은 그룹)"에 속해 있는지 빠르게 판별하기 
* - 대표 활용
  - 그래프의 사이클 판별, 네트워크 연결 여부, MST (최소 신장 트리) (Kruskal 알고리즘)
```

**기본 아이디어** 
- 각 노드는 자기 자신을 부모로 가지는 트리 형태로 시작한다. 
- 집합을 병할할때는 **루트 노드(대표자)** 를 기준으로 합친다. 

```python
# 부모 노드 정보 저장 
parent = [i for i in range(n+1)]

# find 연산 (경로 압축)
def find(x):
  if parent[x] != x:
    parent[x] = find(parent[x])
  return parent[x]

# union 연산 (대표자 기준으로 합치기)
def union(a, b):
  rootA, rootB = find(a), find(b)
  if rootA != rootB:
    parent[rootB] = rootA 
```

### 경로 압축 (Path Compression)

find() 연산을 반복할수록 트리 깊이가 깊어지면 성능이 나빠진다. 
-> 경로 압축을 적용하면 ***모든 노드가 바로 루트를 가리키게 만들어*** 평균 시간복잡도 O($\alpha(N)$)*로 줄일 수 있다. 

* $\alpha$는 아커만 함수의 역함수로 사실상 상수 

예를 들어, union(a, b)로 두 집합이 합쳐진 후에, find함수로 root를 찾으면, parent가 모두 root으로 되어 후에 다시 parent를 찾고자 할때 빠른 시간안에 찾을 수 있다. 

```python
# without compression 
# def find(x):
#   return parent[x]

def find(x):
    if parent[x] != x:
        parent[x] = find(parent[x])  # 루트를 재귀적으로 찾으며 압축
    return parent[x]
```

### 랭크 기반 합치기 (Union by Rank / Size)

두 집합의 크기 (혹은 깊이)를 기준으로 더 작은 트리를 큰 트리에 붙이면 트리의 깊이가 커지지 않아 효율적이다. 

```python
rank = [1]*(n+1)

def union(a, b):
    rootA, rootB = find(a), find(b)
    if rootA == rootB:
        return
    if rank[rootA] < rank[rootB]:
        parent[rootA] = rootB
    elif rank[rootA] > rank[rootB]:
        parent[rootB] = rootA
    else:
        parent[rootB] = rootA
        rank[rootA] += 1
```

```{list-table} union/find 시간 복잡도 
:widths: 15 40 45
:header-rows: 1 

* - 연산 
  - 평균 시간복잡도 
  - 설명 
* - find()
  - O($\alpha (N)$)
  - 경로 압축 사용 시 거의 상수 
* - union()
  - O($\alpha (N)$)
  - find를 포함하므로 동일 
* - 전체 N개의 원소, M개의 연산 
  - O((N+M) $\alpha (N)$) ~ O(N+M)
  - 거의 선형 시간 
```


### 대표 활용 예시 

1. 사이클 판별 (Cycle Detection): 그래프 간선을 순서대로 확인하며 같은 집합에 속한 두 노드를 다시 연결하려 할 때 -> 사이클 존재 
2. 

## 신장 트리 

## 위상 정렬 

