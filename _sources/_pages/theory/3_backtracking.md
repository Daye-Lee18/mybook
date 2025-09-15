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

# Lecture 3-1. Backtracking 

**Backtracking** is a technique where we explore possible candidates step by step, and whenever a candidate does not satisfy the condition, we immediately backtrack and try another option.  
Unlike brute-force, which explores *all* possibilities until the end, backtracking **prunes** the search tree as soon as it finds an invalid state, thus reducing the search space.  

It is similar to **DFS (Depth-First Search)** and can be implemented using recursion and stack structures.  
The goal of backtracking is either to generate all possible solutions or to find a solution that meets certain conditions.  
The basic principle of backtracking is:  

**Choose → Explore → Unchoose (Backtrack)**

````{admonition} Principle of Backtracking
:class: tip
Step 1. **Choose**: At the current stage, add a candidate to the path  
Step 2. **Explore**: Proceed to the next step with a recursive call  
Step 3. **Unchoose / Backtrack**: Remove the last chosen candidate from the path (pop)  
````

For example, backtracking usually follows the structure below:

````{code-block} python 
def backtracking(start):
    for i in range(start, len(arr)):
        path.append(arr[i])          # Choose: select current index
        backtracking(i + 1)          # Explore: recurse to the next index
        path.pop()                   # Unchoose: restore the state
````

Without `pop` (unchoose), it is impossible to prevent duplicate solutions or correctly restore the state.

```{image} ../../assets/img/backtracking/1.png 
:alt: backtracking diagram
:class: bg-primary mb-1
:width: 400px
:align: center
```

## Combination (조합)

Combination means selecting a fixed number of elements (`depth = k`) without considering order.
The key idea: different orders of the same elements are considered the same, and we must handle duplicates carefully.

### Combination without repetition (중복 불허)

Example: choosing 2 cards out of `[1, 2, 3]`.

Since order does not matter, `indices smaller than the current index` are not revisited (to avoid duplicates).
For instance, when index=1, `[1, 2]` is already generated, so `[2, 1]` should not be repeated.

```{image} ../../assets/img/backtracking/2.png
:alt: combination without repetition
:class: bg-primary mb-1
:width: 400px
:align: center
```

```{code-block} python
---
caption: combination without repetition
---
# Combination (순서 무관, 중복 불허, 길이 k)
array = [1, 2, 3]
k = 2
path = []

def combination(start, depth):
    if depth == k:
        print(path)
        return
    for i in range(start, len(array)):   
        path.append(array[i])                   # 선택 (Choose)
        combination(i + 1, depth + 1)           # 다음 원소 탐색 (Explore)
        path.pop()                              # 상태 복원 (Unchoose)

print(f"Combination result of {array} with k={k}:")
combination(0, 0)
print()
```

```{code-block} text
---
caption: The result of combination without repetition
---
Combination without repetition result of [1, 2, 3] with k=2:
[1, 2]
[1, 3]
[2, 3]
```

### Combination with repetition (중복 허용)

If repetition is allowed, call `combination(i, depth+1)` instead of combination(i+1, depth+1).

```{code-block} text
---
caption: The result of combination with repetition
---
Combination result of [1, 2, 3] with k=2:
[1, 1]
[1, 2]
[1, 3]
[2, 2]
[2, 3]
[3, 3]
```

저번 시간에 배웠듯이, 조합 역시 DFS 기반으로 재귀 호출을 반복하며, 내부적으로는 stack 구조로 동작한다. 아래 그림처럼 함수가 호출될 때 스택에 쌓이고, 종료되면 스택에서 빠져나오면서 상태가 되돌려진다. 

```{image} ../../assets/img/backtracking/3.png
:alt: stack for DFS
:class: bg-primary mb-1
:width: 400px
:align: center
```

## Multiset (중복 원소가 있는 집합)

A `multiset` allows duplicates in the input but ensures no duplicate outputs.

Example: for `[1, 1, 2]`, `[1, 1]` is valid, but we should not output `[1, 1]` twice.

즉, **입력 원소의 중복은 허용**되지만, **출력 결과의 중복은 제거**해야 한다.  

Key ideas:
- Sort the array so duplicates are adjacent
- Skip the same value within the same depth level (using if `i > start and arr[i] == arr[i-1]: continue`)

This is the same as [Subset II](https://leetcode.com/problems/subsets-ii/description/).

```{code-block} python
---
caption: Multiset Implementation
---


arr = [1, 2, 1]


def multiset(start):
    arr.sort()
    res, path = [], []
    
    def multiset_helper(start):
        
        res.append(path[:])
        for n_idx in range(start, len(arr)):
            '''
            이 경우는 path를 계속 이어가기 위해 반드시 한 번은 선택해줘야 해요. 즉, 같은 값이 있어도 그 중 첫 번째 등장은 허용해야 새로운 subset이 생겨요. 그래서 조건에 i > start가 붙은 거예요. (첫 번째 원소는 쓰고, 이후 중복만 건너뛰기)
            arr가 increasing order로 되어있으므로, 다음 depth에서 이 전 Index가 가리키는 값과 같은 값을 넣으면 [1] == [1] 이렇게 같아지므로, 피해야한다.
            '''
            if n_idx > start and arr[n_idx] == arr[n_idx-1]:
                continue 
            
            path.append(arr[n_idx])
            # print(f"Call at depth {n_idx}")
            multiset_helper(n_idx+1)
            # print(f"End at depth {n_idx}")
            path.pop()
    
    multiset_helper(0)
    print(res)
        

multiset(arr)

```
```{code-block} text
---
caption: Multiset 결과 
---
[[], [1], [1, 1], [1, 1, 2], [1, 2], [2]]
```

## Subset (Powerset)

Subset means finding`all subsets of a set`.
The collection of all subsets is called the Powerset, with $2^n$ subsets. A set originally has no duplicate elements, and the collection of all its subsets is called the powerset. The powerset can be easily generated by considering two cases for each element: “choose” or “do not choose.” In other words, each element creates a branch, resulting in a total of $2^n$ possible subsets.

---
```{code-block} python
---
caption: Powerset 
---
# ========= 1) Subset (powerset) — 인덱스 기준(중복 값이 있으면 같은 값 모양이 중복 출력될 수 있음)
array = [1, 1, 3]
path = []

def powerset(i):
    if i == len(array): # 모든 원소를 다 확인한 경우
        print(path)
        return

    # 1) 현재 원소 선택하지 않음
    powerset(i + 1)
    # 2) 현재 원소 선택함
    path.append(array[i])
    powerset(i + 1)
    path.pop() # 상태 복원

print(f"Subset (by index) Result of {array}:")
powerset(0)
print()

```

```{code-block} text
Subset (by index) Result of [1, 1, 3]:
[]
[3]
[1]
[1, 3]
[1]
[1, 3]
[1, 1]
[1, 1, 3]
```

One important point is that if the input is a set (i.e., contains no duplicates), then all generated subsets will be unique. However, if the input contains duplicate elements, such as [1, 1, 3], the same subset form can appear multiple times (e.g., [1] and [1]). In such cases, duplicate removal must be handled explicitly, which leads to the Multiset (Subset II) problem.

## Substring & Subarray 

- **Substring**: a contiguous part of a string
- **Subarray**: a contiguous part of an array (list)

In both cases, order matters, and they must be contiguous. The only difference is that substrings come from strings, while subarrays come from numerical arrays; conceptually, they are the same. These problems do not require DFS or backtracking and can be implemented using two nested for-loops.

- Outer loop: starting index (start pointer)
- Inner loop: length or ending index (end pointer)

```{code-block} python
# --- Substring: 문자열의 연속 부분문자열 전부
my_str = "abc"
def all_substrings(s):
    res = []
    n = len(s)
    for s in range(n): # 시작 인덱스
        for t in range(s+ 1, n + 1):  # 끝 인덱스 (exclusive), [i, j)
            res.append(my_str[s:t])
    return res

print(f"All substrings of '{my_str}':")

for sub in all_substrings(my_str):
    print(sub)
print()
```

```{code-block} text 
All substrings of 'abc':
a
ab
abc
b
bc
c
```

```{code-block} python
---
caption: 모든 Subarray 구하기
---
arr = [1, 2, 3]

def all_subarrays(arr):
    res = []
    n = len(arr)
    for i in range(n):               # 시작 인덱스
        for j in range(i+1, n+1):    # 끝 인덱스 (exclusive)
            res.append(arr[i:j])
    return res

print(f"All subarrays of {arr}:")
for sub in all_subarrays(arr):
    print(sub)

```

```{code-block} text 
All subarrays of [1, 2, 3]:
[1]
[1, 2]
[1, 2, 3]
[2]
[2, 3]
[3]
```

## Subsequences 

A subsequence is formed from an existing array (or string) where the order of elements must be preserved, but they do **not need to be contiguous**.
  
- Subsequences of the string "abc":
    - "a", "b", "c", "ab", "ac", "bc", "abc", "" (including the empty set)
    - "ac" is a subsequence but not a substring (because it is not contiguous).

Backtracking (DFS) is used to explore subsequences by choosing or skipping each element. Therefore, the total number of subsequences is $2^n$. Whether to include the empty set can be adjusted depending on the problem. In terms of implementation, subsequences are very similar to combinations; the difference is that combinations have a fixed target depth, whereas subsequences proceed from the starting index and output all possible paths of lengths from 0 to n.

```{code-block} python
---
caption: 모든 Subsequence 구하기
---
array = [1, 2, 3]
path = []

def subsequences(start):
    print(path)  # 공집합 포함 (필요 없으면 if start>0일 때만 출력 등으로 조절)
    for nxt in range(start, len(array)):
        path.append(array[nxt]) # 원소 선택
        subsequences(nxt + 1)  # 다음 인덱스로 진행 ✅ 인덱스 증가가 포인트
        path.pop() # 선택 취소 (백트래킹)

print(f"Subsequences Result of {array}:")
subsequences(0)
print()
```

```{code-block} text
Subsequences Result of [1, 2, 3]:
[]
[1]
[1, 2]
[1, 2, 3]
[1, 3]
[2]
[2, 3]
[3]
```

## Permutation (순열)

A permutation, denoted as ${n}P{k}$, refers to the number of ways to choose $k$ elements from $n$ elements while considering order. Similar to combinations, it has a target depth of $k$, but since order matters, $(1, 2)$ and $(2, 1)$ are recognized as different results.

```{code-block} python 
# 길이 k, 중복 불허 → 방문 체크 필요
array = [1, 2, 3]
k = 2
path = []
used = [False] * len(array)

def permutation_no_repeat(depth):
    if depth == k:
        print(path)
        return
    for i in range(len(array)):
        if used[i]:                 # 이미 쓴 원소는 건너뜀
            continue
        used[i] = True              # 사용 처리
        path.append(array[i])       # 후보 선택 (Choose)
        permutation_no_repeat(depth + 1)  # 다음 단계 탐색 (Explore)
        path.pop()                  # 상태 복원 (Unchoose)
        used[i] = False             # 다시 미사용 처리

print(f"Permutation (no repetition) of {array} with k={k}:")
permutation_no_repeat(0)
```

```{code-block} python
# 길이 k, 중복 허용 → product와 동일
array = [1, 2, 3]
k = 2
path = []

def k_tuples_with_repetition(depth):
    if depth == k:
        print(path)
        return
    for x in array:                 # 모든 원소를 다시 선택 가능
        path.append(x)              # 후보 선택 (Choose)
        k_tuples_with_repetition(depth + 1)  # 다음 단계 탐색 (Explore)
        path.pop()                  # 상태 복원 (Unchoose)

print(f"k-tuples with repetition (a.k.a. product) of {array} with k={k}:")
k_tuples_with_repetition(0)

```

## 정리 
```{admonition} Summary 
:class: important 
- **Combination**: order does not matter, with a fixed length $k$.  
  - Without repetition: `combination(i+1, depth+1)`  
  - With repetition: `combination(i, depth+1)`  
- **Multiset**: a subset problem with duplicate inputs, solved by sorting + skipping duplicates at the same depth.  
- **Powerset**: finding all subsets, with $2^n$ total, implemented by the binary choice of “select or not select” for each element.  
- **Substring/Subarray**: only contiguous parts are allowed. The number of such cases is always $n(n+1)/2$, and they can be generated easily with a double for-loop.  
- **Subsequence**: order must be preserved, but contiguity is not required. Implementation is similar to combination, except that the depth is not fixed.  
- **Permutation**: generated with DFS by exploring all indices starting again from index 0 at each depth.  
```
## 연습 문제 

### Subset II - Multiset (중복 입력 처리)
문제 - [leetcode 90번 문제](https://leetcode.com/problems/subsets-ii/description/)

### Permutations 관련 문제
문제 - [leetcode 46번](https://leetcode.com/problems/permutations/description/)

### Combination 관련 문제
문제 - [백준 1759 암호 만들기](https://www.acmicpc.net/problem/1759)

### Permutations 관련 문제
문제 - [프로그래머스 42839 소수찾기](https://school.programmers.co.kr/learn/courses/30/lessons/42839)

### Combination 관련 문제
문제 - [프로그래머스 84512 모음 사전](https://school.programmers.co.kr/learn/courses/30/lessons/84512)