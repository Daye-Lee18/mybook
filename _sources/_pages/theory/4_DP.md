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

# Lecture 4-1. Dynamic Programming 

컴퓨터는 연산 속도에 한계가 있고, 메모리 공간을 사용할 수 있는 데이터의 개수도 한정적이다. 따라서, 연산 속도와 메모리 공간을 최대한 효율적으로 활용하는 알고리즘을 작성해야한다. 다만, **어떤 문제는 메모리 공간을 약간 더 사용하면 연산 속도를 비약적으로 증가**시킬 수 있는 방법이 있다. 대표적으로 이번 장에서 다루는 다이나믹 프로그래밍 (Dynamic Programming, 동적 계획법)기법이다. 

## DP 조건 

DFS를 배울 때, 피보나치 수열을 구현하는 방법을 배웠다. 피보나치 수열은 다음 그림과 같은 형태로 끝없이 이어진다. 

![1](../../assets/img/DP/1.png)

점화식을 통해 인접한 항들 사이의 관계식을 의미하는데, 예를 들어 수열 ${a_{n}}$이 있을 때 수열에서의 각 항을 $a_{n}$이라고 부른다고 가정하자. 우리는 점화식을 통해 현재의 항을 이전의 항에 대한 식으로 표현할 수 있다. 예를 들어, 피보나치 수열의 점화식은 다음과 같이 표현할 수 있다. 

$a_{n+2} = a_{n} + a_{n+1}, a_{1} =1, a_{2} = 2$

위와 같은 점화식은 인접 3항간 점화식이라고 부르는데, 인접한 총 3개의 항에 대해서 식이 정의되기 때문이다. 

프로그래밍에서는 이러한 수열을 **배열**이나 **리스트**로 표현할 수 있다. 수열 자체가 여러 개의 수가 규칙에 따라서 배열된 형태를 의미하는 것이기 때문이다. 파이썬에서는 리스트 자료형이 이를 처리하고, C/C++ 와 자바에서는 배열을 이용해 이를 처리한다. 

위와 같은 점화식을 실제로 어떻게 구현할 수 있을까? n번째 피보나치 수를 f(n)이라고 표현할 때 4번째 피보나치 수 f(4)를 구하려면 다음과 같이 함수 f를 반복해서 호출할 것이다. 그런데 f(2)와 f(1)은 항상 1이기 때문에 f(1)이나 f(2)를 만났을 때는 호출을 정지한다. 

![2](../../assets/img/DP/2.png)

수학적 점화식을 프로그래밍으로 표현하려면 재귀 함수를 사용하면 간단하다. 예시를 소스코드로 바꾸면 다음과 같다. 

````{toggle}
```{code-block} python
# 피보나치 함수(Fibonacci Function)을 재귀함수로 표현 
def fibo(x):
  if x == 1 or x == 2:
    return x 

  return fibo(x-1) + fibo(x-2)

print(fibo(4))
```
````

피보나치 수열의 소스코드를 위와 같이 작성하면 문제가 생길 수 있는데, f(n) 함수에서 n이 커지면 커질수록 수행 시간이 기하급수적으로 늘어나기 때문이다. 그림을 보면 동일한 함수가 반복적으로 호출되는 것을 알 수 있다. 이미 한 번 계산했지만, 계속 호출할 때마다 계산하는 것이다. 이처럼 피보나치 수열의 점화식을 재귀 함수를 사용해 만들 수 는 있지만, 단순히 매번 계산핟로로고 하면 문제를 효율적으로 해결할 수 없다. 이러한 문제는 다이나믹 프로그래밍을 사용하면 효율적으로 해결할 수 있다. 하지만 다이나믹 프로그래밍을 항상 사용할 수는 없으며, 다음 조건을 만족할 때 사용할 수 있다. 

```{admonition} DP 문제 조건 
:class: important 
1. 큰 문제를 작은 문제로 나눌 수 있다. 
2. 작은 문제에서 구한 정답은 그것을 포함하는 큰 문제에서도 동일하다. 
```

피보나치 수열은 위의 조건을 만족하는 대표 문제로, 이 문제를 메모이제이션 (Memoization) 기법을 사용해서 해결해보자. 메모이제이션은 다이나믹 프로그래밍을 구현하는 방법 중 한 종류로, 한 번 구한 결과를 메모리 공간에 메모해두고 같은 식을 다시 호출하면 메모한 결ㄹ과를 그대로 가져오는 기법을 의미한다. 메모이제이션은 값을 저장는 방법이므로 캐싱 (Caching)이라고도 한다. 

메모이제이션의 구현은 단순히, 한 번 구한 정보를 리스트에 저장하는 것이다. 다이나믹 프로그래밍을 재귀적으로 수행하다가 같은 정보가 필요할 때는 이미 구한 정답을 그대로 리스트에서 가져오면 된다. 

````{toggle}
```{code-block} python 
# 한 번 계산된 결과를 메모이제이션(Memoization)하기 위한 리스트 초기화
d = [0] * 100 

# 피보나치 함수 (Fibonacci Function) 를 재귀함수로 구현 (탑다운 다이나믹 프로그래밍)
def fibo(x):
    # base case 
    if x == 1 or x == 2:
        return 1 
    
    # 이미 계산한 적 있는 문제라면 그대로 반환 
    if d[x] != 0:
        return d[x] 
    
    # 아직 계산하지 않은 문제라면 점화식에 따라서 피보나치 결과 반환 
    d[x] = fibo(x-1) + fibo(x-2)
    return d[x]

print(fibo(99))
```
````

## DP 2가지 구현 방식

정리하자면, 다이나믹 프로그래밍이란 큰 문제를 작게 나누고, 같은 문제라면 한 번씩만 풀어 문제를 효율적으로 해결하는 알고리즘 기법이다. 다이나믹 프로그래밍과 분할 정복 (Divide and Conquer)의 차이점은 다이나믹 프로그래밍은 문제들이 서로 영향을 미치고 있다는 점이다. 
즉, 둘 다 큰 문제를 작은 문제로 쪼개 푸는 건 비슷하지만, 차이는 부분 문제의 관계에 있습니다. 

- 분할 정복 (예: 퀵정렬, 병합정렬)
  - 문제를 작은 문제로 나눔 → 각각 독립적으로 해결 → 결과를 합쳐서 큰 문제 해결.
  - 각 부분 문제는 서로 영향을 주지 않음.
  - 예시: merge sort에서 왼쪽 배열 정렬과 오른쪽 배열 정렬은 서로 간섭 없음.

- 다이나믹 프로그래밍 (예: 피보나치, 최단 경로)
  - 문제를 작은 문제로 나눔 → 작은 문제들의 해가 서로 겹치거나 공유됨.
  - 한 번 계산한 결과를 저장해놔야 효율적.
  - 예시: 피보나치 수열을 단순 분할 정복으로 풀면 같은 계산을 수없이 반복하지만, DP로 풀면 한 번 계산한 값을 저장해서 재사용함.

```{admonition} DP 개념 정리 
:class: note 
1.큰 문제를 작은 문제로 나눔: 문제를 세분화해서 풀기 
1. 중복되는 작은 문제는 "한 번만" 계산: 계산이 반복되면 결과를 저장 (메모이제이션/테이블화)해서 재사용 

즉, "중복되는 부분 문제 (overlapping subproblems)"를 효율적으로 처리하는 방식
```

아래 그림을 보면, f(6)을 호출할 때는 다음 그림처럼 색칠된 노드만 방문하게 되어, 효율적으로 문제를 풀 수 있다. 

![3](../../assets/img/DP/3.png)

## 메모이제이션 

메모이제이션은 때에 따라서 다른 자료형, 예를 들어 사전 (Dict) 자료형을 이용할 수도 있다. 사전 자료형은 수열처럼 `연속적이지 않은 경우`에 유용한데, 예를 들어 $a_{n}$을 계산하고자 할 때 $a_{0} ~ a_{n-1}$모두가 아닌 일부의 작은 문제에 대한 해답만 필요한 경우가 존재할 수 있다. 이럴 때에는 사전 자료형을 사용하는게 더 효과적이다. 

또한 가능하다면 재귀 함수를 이용하는 탑다운 방식보다는 보텀업 방식으로 구현하는 것을 권장한다. 시스템상 재귀 함수의 스택 크기가 한정되어 있을 수 있기 때문이다. 이 경우 sys 라이브러리에 포함되어 있는 `setrecursionlimit()` 함수를 호출하여 재귀 제한을 완화할 수 있다는 점만 기억하자. 

## 시간복잡도 

피보나치 수열 알고리즘의 시간 복잡도는 O(N)이다. 왜냐하면 f(1)을 구한 다음 그 값이 f(2)를 푸는 데 사용도기, f(2)의 값이 f(3)를 푸는데 사용되는 방식으로 이어지기 때문이다. 한 번 구한 결과는 다시 구해지지 않는다. 

### Top-Down 방식 

이처럼 재귀 함수를 이용하여 다이나믹 프로그래밍 소스 코드를 작성하는 방법을, 큰 문제를 해결하기 위해 작은 문제를 호출한다고 하여 top-down 방식이라 말한다. 

### Bottom-Up 방식 

단순히 반복문을 이용하여 소스 코드를 작성하는 경우 작은 문제부터 차근차근 답을 도출한다고 하여 Bottom-Up 방식이라고 말한다. 


````{toggle}
```{code-block} python
---
caption: 피보나치 수열 소스코드 (Bottom-Up)
---
# 앞서 계산된 결과를 저장하기 위한 DP 테이블 초기화 
d = [0] * 100 

# 첫 번째 피보나치 수와 두 번째 피보나치 수는 1 
d[1] = 1 
d[2] = 1
n = 99

# 피보나치 함수 반복문으로 구현 
for i in range(3, n+1):
    d[i] = d[i-1] + d[i-2]

print(d[n])

```
````

## 예시 문제 

### 1. 1D DP 

#### Climbing Stairs 
문제 - [Leetcode 70](https://leetcode.com/problems/climbing-stairs/?envType=study-plan-v2&envId=top-interview-150)

#### House Robber 
문제 - [Leetcode 198](https://leetcode.com/problems/house-robber/description/?envType=study-plan-v2&envId=top-interview-150)

#### Word Break
문제 - [Leetcode 139](https://leetcode.com/problems/word-break/description/?envType=study-plan-v2&envId=top-interview-150)

#### Coin Change 
문제 - [Leetcode 322](https://leetcode.com/problems/coin-change/description/?envType=study-plan-v2&envId=top-interview-150)

#### Longest Increasing Subsequences 
문제 - [Leetcode 300](https://leetcode.com/problems/longest-increasing-subsequence/?envType=study-plan-v2&envId=top-interview-150)

### 2. Multidimensional DP 

#### Triangle 
문제 - [Leetcode 120](https://leetcode.com/problems/triangle/description/?envType=study-plan-v2&envId=top-interview-150)

#### Minimum Path Sum
문제 - [Leetcode 64](https://leetcode.com/problems/minimum-path-sum/description/?envType=study-plan-v2&envId=top-interview-150)

#### Unique Path II
문제 - [Leetcode 63](https://leetcode.com/problems/unique-paths-ii?envType=study-plan-v2&envId=top-interview-150)

#### Longest Palindromic Substring 
문제 - [Leetcode 5](https://leetcode.com/problems/longest-palindromic-substring/description/?envType=study-plan-v2&envId=top-interview-150)

#### Interleaving String 
문제 - [Leetcode 97](https://leetcode.com/problems/interleaving-string/description/?envType=study-plan-v2&envId=top-interview-150)

#### Edit Distance 
문제 - [Leetcode 72](https://leetcode.com/problems/edit-distance/description/?envType=study-plan-v2&envId=top-interview-150)

