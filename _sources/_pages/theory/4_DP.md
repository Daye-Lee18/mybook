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
1. 큰 문제를 작은 문제로 나눔: 문제를 세분화해서 풀기 
2. 중복되는 작은 문제는 "한 번만" 계산: 계산이 반복되면 결과를 저장 (메모이제이션/테이블화)해서 재사용 

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

````{toggle}

Bottom-up 방식으로 쉽게 풀 수 있다. 현재 선택지는 +1, +2칸이므로 f(n)을 계산하는 경우 n-1 칸에서 한 칸 오른 것 즉, f(n-1)과 n-2칸 에서 +2칸 오른 것 f(n-2)를 더한 것과 동일하다. 

```{code-block} python
class Solution:
    def climbStairs(self, n: int) -> int:
        '''
        1 = 1

        N=2 
        1 + 1 
        2 

        N = 3 
        1 + 1 + 1 
        1 + 2 
        2 + 1 
        
        N = 4 
        1 + 1 + 1 + 1 
        1 + 2 + 1 
        2 + 1 + 1 
        1 + 1 + 2 
        2 + 2 

        즉, f(N) = f(N-1) + f(N-2)
        f(N-1)에서는 1만 더하면 되고 f(N-2)는 2를 더하면 됨. 
        '''
        dp = [0] * (n+1)
        if n <=2:
            return n
        dp[1] = 1
        dp[2] = 2
        for i in range(3, n+1):
            dp[i] = dp[i-1] + dp[i-2]

        return dp[n]

if __name__ == '__main__':
    s = Solution()
    print(s.climbStairs(3))
```
````
#### House Robber 
문제 - [Leetcode 198](https://leetcode.com/problems/house-robber/description/?envType=study-plan-v2&envId=top-interview-150)

````{toggle}
```{code-block} python 
from typing import List 

class Solution:
    def rob(self, nums: List[int]) -> int:
        n = len(nums)

        if n == 1:
            return nums[0]
        
        dp = [0] * n

        
        dp[0] = nums[0]
        dp[1] = max(nums[0], nums[1]) # 해당 index까지 최고 dp 값 저장 
        max_val = dp[1]
        for idx in range(2, n):
            dp[idx] = max(dp[idx-2] + nums[idx], dp[idx-1]) # 해당 index까지 최고 dp 값 저장 
            max_val = max(max_val, dp[idx])

        return max_val
        
        
if __name__ == '__main__':
    # nums = [10, 1 , 2 , 3] # 13 
    # nums = [10, 1 , 2 , 3, 12] # 24 
    # nums = [1, 2, 3, 1] # 4
    # nums = [2, 7, 9, 3, 1] # 12
    nums = [1, 3, 1, 3, 100] # 103

    s = Solution()
    print(s.rob(nums))

```
````

#### Word Break
문제 - [Leetcode 139](https://leetcode.com/problems/word-break/description/?envType=study-plan-v2&envId=top-interview-150)

````{toggle} 
## Bottom-Up 

아이디어: 
- `dp[i] = s[:i]` 가 사전 단어들로 정확히 분해 가능하면 True.
- 시작 dp[0] = True (빈 문자열은 분해 가능)
- 각 i에 대해, 어떤 j < i가 있어 dp[j] == True 이고 s[j:i] 가 단어면 dp[i] = True.

효율 팁
- 단어들을 set으로 만들어 O(1) 조회.
- 단어 길이의 최대값 L = max(len(w) for w in wordDict)를 구해 j를 i-L .. i-1로만 확인(큰 가지치기).
```{code-block} python 

from typing import List 

class Solution:

    def wordBreak(self, s: str, wordDict: list[str]) -> bool:
        word_set = set(wordDict)
        max_len = max(map(len, word_set)) if word_set else 0
        n = len(s)
        dp = [False] * (n + 1)
        dp[0] = True

        for r in range(1, n + 1):
            # r에서 최대 max_len만큼만 거슬러 올라가 보기
            start = max(0, r - max_len)
            for f in range(r - 1, start - 1, -1):
                if dp[f] and s[f:r] in word_set:
                    dp[r] = True
                    break
        return dp[n]

            
        
if __name__ == '__main__':
    # s = "abc"; wordDict=["a", "b", "c"]
    # s = "leetcode"; wordDict = ["leet", "code"]
    # s = "applepenapple"; wordDict = ["apple", "pen"]
    # s = "catsandog"; wordDict = ["cats", "dog", "sand", "and", "cat"]
    s = "catsand"; wordDict = ["cats", "dog", "sand"]

    sol = Solution()
    print(sol.wordBreak(s, wordDict))

```
````

#### Coin Change 
문제 - [Leetcode 322](https://leetcode.com/problems/coin-change/description/?envType=study-plan-v2&envId=top-interview-150)

````{toggle}
```{code-block} python
---
caption : Greedy로 찾을 수 없다. 가장 큰 동전을 일단 뽑는 선택이, 이후 남은 금액을 더 나쁘게 만들어 전체 개수를 늘려버릴 수 있어요. 예를 들어, coins = [1, 5, 11], amount = 15 일때, 그리디: 11 → (남은 4) → 1×4 ⇒ 5개, 정답인 최적: 5 + 5 + 5 ⇒ 3개 이다. Greedy가 통하는 경우는 동전 체계가 `canonical(정규)`한 경우에 가능하다. 예를 들어, 각 동전이 이전 동전의 배수인 경우 (예: [1, 2, 4, 8..])
---
from typing import List 

class Solution:

    def in_range(self, cur_idx, dif):
        return (cur_idx - dif) > 0
    
    def coinChange(self, coins: List[int], amount: int) -> int:
        
        MAX = 99999999
        dp = [99999999] * (amount+1)
        dp[0] = 0 # basecase, when amount = 0

        for cur_dp_idx in range(1, amount+1):
            for coin in coins:
                if cur_dp_idx - coin == 0: # 현재 idx에 해당하는 coin이 있으면 무조건 1개로 끝낼 수 있음. 
                    dp[cur_dp_idx] = 1 
                if self.in_range(cur_dp_idx, coin):
                    # 가지고 있는 코인 '하나' 써서 만든 거 (dp[cur_dp_idx-coin] + 1)랑 이전 dp (dp[cur_dp_idx])랑 비교 
                    dp[cur_dp_idx] = min(dp[cur_dp_idx], dp[cur_dp_idx-coin]+1)
        # print(dp)
        return dp[amount] if dp[amount] != MAX else -1 

if __name__ == "__main__":
    # coins = [1, 2, 5, 10]; amount = 11  # 2
    # coins = [1, 2, 5, 10]; amount = 11  # 3
    # coins = [2]; amount = 3 
    coins = [1]; amount = 0
    # coins = [1, 2, 5]; amount = 20
    # coins = [186,419,83,408]; amount = 6249 # 20개 
    sol = Solution()
    print(sol.coinChange(coins, amount))
```
````

#### Longest Increasing Subsequences 
문제 - [Leetcode 300](https://leetcode.com/problems/longest-increasing-subsequence/?envType=study-plan-v2&envId=top-interview-150)

````{toggle}
##  bisect library, 위치 찾기 in LIS problem

`bisect`는 파이썬 표준 라이브러리로, 정렬된 리스트에서 이진 탐색을 통해 원소를 삽입하거나 위치를 찾을 때 사용합니다. 
정렬은 미리 되어 있어야 하고, 삽입 위치를 계산해줄 뿐 자동으로 정렬을 유지하지는 않습니다. 

복잡도: O(logN)

1. bisect_left(arr, x)
- 정렬된 리스트 arr에서 x가 들어갈 가장 왼쪽 인덱스를 반환. 
- 즉, x 이상의 원소가 처음 나타나는 위치 

```{code-block} python 
import bisect

a = [1, 2, 4, 4, 8]
print(bisect.bisect_left(a, 4))  # 2 (왼쪽 4의 위치)
print(bisect.bisect_left(a, 5))  # 4 (8이 있는 자리)
```
2. bisect_right(arr, x)
- 정렬된 리스트 arr에서 x가 들어갈 가장 오른쪽 인덱스를 반환
- 즉, x보다 큰 원소가 처음 나타나는 위치 

```{code-block} python
print(bisect.bisect_right(a, 4)) # 4 (오른쪽 4 다음 위치)
print(bisect.bisect(a, 4))       # 동일 (alias)

```

복잡도: O(logN) (탐색) + log(N) (리스트에 삽입, 뒤 원소를 밀어야 함.)


3. insort_left(arr, x)
- bisect_left로 위치를 찾은 뒤 x를 리스트에 삽입 
- 정렬된 상태 유지 

```{code-block} python
bisect.insort_left(a, 4)
print(a)  # [1, 2, 4, 4, 4, 8]
```
4. insort_right(arr, x)
- bisect_right로 위치를 찾은 뒤 x를 리스트에 삽입 
- 정렬된 상태 유지 

```{code-block} python
b = [1, 2, 4, 4, 8]
bisect.insort_right(b, 4)
print(b)  # [1, 2, 4, 4, 4, 8]
```
````

````{toggle} 

```{code-block} python
---
caption: 시간 복잡도 O(NlogN)으로 푸는 전형적인 LIS 알고리즘은 "patience sorting (꼭대기 배열 tails 유지) + 이분 탐색"이다. 즉, tails[k] = "길이가 k+1인 증가 부분수열 중 마지막 값의 최소값" 을 저장한다. 배열을 왼 -> 오 순서로 보며, 매 원소에 대해 `i = lower_bound(tails, x)` (즉, tails에서 x이상이 처음 나오는 위치)를 찾고 i == len(tails)면 tails뒤에 x를 붙이고, 아니면 tails[i] = x로 더 작은 끝값으로 갱신한다. 이렇게 하면 tails의 길이가 곧 LIS의 길이가 됨. 만약 비내림 (= 증가 허용, non-decreasing)인 경우에는 `bisect_right`을 사용하여 같은 값은 다음 길이로 넘겨 중복을 허용한다. 
---
import bisect

class Solution:
    def lengthOfLIS(self, nums: List[int]) -> int:
        tails = [] # 길이가 idx+1인 수의 가장 작은 수를 저장 

        for num in nums:
            # strictly increasing 
            idx = bisect.bisect_left(tails, num)
            if idx == len(tails):
                tails.append(num)
            else:
                tails[idx] = num 

        return len(tails)

if __name__ == "__main__":
    nums = [10, 9, 2, 5, 3, 7, 101, 18]
    # nums = [0, 1, 0, 3, 2, 3] # 4 
    # nums = [7, 7, 7] # 1
    # nums = [7] # 1
    sol = Solution()
    print(sol.lengthOfLIS(nums))

```
````

`````{admonition} bisect_left implementation
:class: dropdown

- **Boundary definition with `while l < r` (Step 1)**  
    - Always maintain the interval [l, r) (left-inclusive, right-exclusive)  
    - `l` is always a candidate position, and `r` is always the boundary (one step beyond the last possible position).  
    - The search terminates when `l == r` → no interval left to search.  
        - `l` = "the minimum candidate position to insert"  
        - `r` = "the first position where insertion is impossible (exclusive right boundary)"  
    - When they become equal, the insertion position is fixed.  
    - Hence the condition must always be `while l < r`.  
    - If we use `while l <= r`, then `r` must be treated as inclusive, and the code logic changes.  

- **Example**  
    - `arr = [2, 5, 6]`, `x = 4`  
    - Initial: `l = 0`, `r = 3`, `mid = 1`  
        - Think of `mid` as the current comparison candidate.  
    - Step 1: `mid = 1`, `arr[mid] = 5 >= 4` → `r = mid`  
        - For `bisect_left`, when the target is equal to or smaller than the midpoint, we shrink the search interval to the left.  
    - Step 2: `mid = 0`, `arr[mid] = 2 < 4` → `l = mid + 1 = 1`  
    - Termination: `l = 1`, `r = 1` → `return l (1)`  
    - Indeed, inserting `4` at `arr[1]` keeps the array sorted.  

````{toggle}
```{code-block} python
---
caption: bisect_left() implementation code
---

def bisect_left(arr, x):
    l = 0, r = len(arr) # r = right-exclusive boundary
    while l < r:
        mid = (l + r) // 2 

        if arr[mid] >= x:
            r = mid 
        else:
            l = mid + 1 
    return l # l = the 'first' insertion position of x 

````

`````


````{toggle}
```{code-block} python
:caption: 수열까지 복원하기. prev_idx로 연결 리스트를 만들어 마지막 인덱스에서 거꾸로 따라가면 됩니다. 

import bisect

def lis_with_path(nums):
    n = len(nums)
    tails = []            # 값 기준(이분탐색용)
    tails_idx = []        # 각 길이의 꼬리 원소가 있는 nums의 인덱스
    prev_idx = [-1] * n   # 각 원소의 LIS에서의 이전 원소 인덱스

    for i, x in enumerate(nums):
        pos = bisect.bisect_left(tails, x)
        if pos == len(tails):
            tails.append(x)
            tails_idx.append(i)
        else:
            tails[pos] = x
            tails_idx[pos] = i
        if pos > 0:
            prev_idx[i] = tails_idx[pos - 1]

    # 복원
    lis_len = len(tails)
    k = tails_idx[-1]
    seq = []
    while k != -1:
        seq.append(nums[k])
        k = prev_idx[k]
    seq.reverse()
    return lis_len, seq
```
````
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

