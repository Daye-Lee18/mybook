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

# Lecture 10-1. Prefix Sum (누적 합)

## 학습 목표

- Prefix(접두)의 의미를 이해한다.  
- 누적 합 배열(Prefix Sum Array)을 직접 구현한다.  
- 누적 합을 활용해 구간 합(range sum)을 빠르게 계산한다.  


## Prefix란?

**Prefix**는 “접두사”라는 뜻으로, 어떤 수열의 **처음부터 특정 위치까지의 부분 구간**을 의미한다.

예를 들어,  배열이 [1, 2, 3, 4]일 때  

| 구간 | 값 | 설명 |
|------|----|------|
| 첫 번째 원소까지의 합 | 1 | (1) |
| 두 번째 원소까지의 합 | 3 | (1 + 2) |
| 세 번째 원소까지의 합 | 6 | (1 + 2 + 3) |
| 네 번째 원소까지의 합 | 10 | (1 + 2 + 3 + 4) |

이렇게 “앞에서부터 누적된 합”을 담은 새로운 배열을 **Prefix Sum Array**라고 부른다.


## Prefix Sum 배열 구하기

```python
# 기본 배열
data = [1, 2, 3, 4, 5]

# prefix_sum[i]: data[0] ~ data[i]까지의 합 (i is included)
prefix_sum = [0] * len(data)
prefix_sum[0] = data[0] # INIT 

for i in range(1, len(data)):
    prefix_sum[i] = prefix_sum[i-1] + data[i]

print("Prefix Sum:", prefix_sum)
```
출력: 
```bash
Prefix Sum: [1, 3, 6, 10, 15]
```

## Prefix Sum을 왜 쓰는가? 구간합

어떤 구간의 합을 구해야할 때, 매번 for문을 돌면 $O(N)$이 걸린다. 하지만 prefix sum을 미리 계산하면 한 번의 뺄셈으로 구간 합을 $O(1)$에 계산 가능하다. 

구간 합 공식은 다음과 같다. <br>

배열 인덱스가 0부터 시작한다고 할 때, (i, j)의 구간합은 

```{math}
$$
\sum(i, j) = prefix[j] - prefix[i-1]
$$

단, $i=0$일 때는 단순히 $\text{prefix}[j]$.
```

```python
data = [10, 20, 30, 40, 50]

# 1. prefix sum 배열 만들기
prefix = [0] * len(data)
prefix[0] = data[0]
for i in range(1, len(data)):
    prefix[i] = prefix[i-1] + data[i]

# 2. 구간 [1, 3]의 합 (20 + 30 + 40)
i, j = 1, 3
range_sum = prefix[j] - prefix[i-1]
print("Range Sum [1,3] =", range_sum)
```
출력:
```bash
Range Sum [1,3] = 90
```

````{admonition} Prefix 배열의 첫 번째 원소를 0으로 두기 
:class: tip

문제 풀이 시에는 계산을 단순하게 하기 위해 보통 prefix 배열을 길이 N+1로 만들고, prefix[0] = 0으로 초기화한다. 
즉, ***prefix[i]에는 인덱스 [0, i-1]까지의 누적합 = 원소 [1, i] 번째 (1-based) 누적합*** 이 저장된다. 

- prefix[1] = prefix[0] + data[0] → data[0]까지의 합
- prefix[2] = prefix[1] + data[1] → data[0] + data[1]까지의 합
- ... 
- prefix[i] = data[0]부터 data[i-1]까지의 합

따라서, 원소 [2, 5]번째까지의 누적합은 인덱스 [1, 4]까지의 누적합이고 따라서, prefix[5]-prefix[2]로 계산하면 된다. 

```{code-block} python
data = [10, 20, 30, 40, 50]
n = len(data)
prefix = [0] * (n + 1) # prefix[0] = data[0] is not needed 

for i in range(1, n + 1):
    prefix[i] = prefix[i-1] + data[i-1]

# 구간 [2, 5]의 합 = prefix[5] - prefix[1]
print(prefix)
print("Sum(2,5) =", prefix[5] - prefix[1])
```
출력:
```bash
[0, 10, 30, 60, 100, 150]
Sum(2,5) = 140
```
````

## 실전 예시 

|문제 유형| 예시 |
|---|---|
| 1D 누적 합| 구간 합, 평균 구하기|
| 2D 누적 합 | 영역의 합 (e.g. 이미지 누적 밝기)|
| Prefix XOR | XOR 구간 연산 문제 | 
| Prefix Min/Max | 부분 최솟값, 최댓값 추적 |
| 문자열 Prefix | 접두사 비교, KMP 전처리 | 

````{admonition} Summary
:class: important 

"Prefix sum은 계산의 중복을 없애는 가장 간단한 전처리 기법이다."

- Prefix Sum은 누적 합 배열로, 반복 계산을 한 번의 뺄셈으로 줄여줌
- 한 번 계산해두면, 구간 합을 $O(1)$에 처리 가능
- 고급 문제(누적 XOR, 구간 평균, DP 전처리)에서도 자주 등장
````

### 2D Prefix Sum 

### Prefix XOR 응용 

## 연습 문제 

### 구간 합 구하기 4 

[Baekjoon 11659]

### 구간 합 구하기 5 
[Baekjoon 11660]

### Subarray Sum Equals K
[LeetCode 560]

### XOR Queries of a Subarray
[LeetCode 1310]
