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

# Lecture 6-1. 정렬 

정렬 (Sorting)이란 데이터를 특정한 기준에 따라서 순서대로 나열하는 것을 말한다. 정렬 알고리즘으로 데이터를 정렬하면 이후에 배울 이진 탐색 (Binary Search)이 가능해진다. 이번 수업에서는 총 4가지 정렬 (선택 정렬, 삽입 정렬, 퀵 정렬, 계수 정렬)에 대해서 배운다. 이 장에서 다루는 예제는 모두 **오름차순 (Increasing order)** 정렬을 수행한다고 가정한다. 내림차순 정렬은 리스트의 원소를 뒤집는 (Reverse) 메서드를 제공하기 때문에 그 방법을 사용하여 O(N) 시간 복잡도로 만들 수 있다.

`````{admonition} Reversing ways in Python 
:class: dropdown 

```{code-block} python 
# method 1 
list1 = [1, 2, 5, 6]
reversed_list = list1[::-1]
print(reversed_list)

# method 2 
list2 = [1, 3, 5, 6]
list2.reverse() # list의 built-in 함수인 reverse()를 사용하면 original data가 없어지는 단점이 존재. 
print(list2) 
```
`````

`````{admonition} slicing a list 
:class: dropdown 

슬라이싱이란, 리스트 아이템의 일부를 자르는 것이다. 형식은 다음과 같다. 

```{admonition} slicing 방법 
:class: note 

리스트변수[start:end:step]
```

|항목 | 의미 | 기본값 step > 0 | 기본값 step < 0 | 중요포인트 | 
|---|---|---|---|---|
|start| 시작인덱스|0 | -1 | 포함됨 | 
|end | 끝인덱스 | len(arr) | -(len(arr)+1) | 포함안됨 | 
|step | 간격 (양수면 오른쪽, 음수면 왼쪽으로 이동) | 1 | 1 | 방향 결정자 | 

- 핵식 개념 정리 
    - start, end, step 모두 생략 가능하다.
    - start와 end의 상대적 방향이 step과 맞아야한다. 
        - 오른쪽으로 갈때는 start < end 여야 값이 생기고, 
        - 왼쪽으로 갈때는 start > end여야 값이 생긴다. 
        - 이 조건이 안 맞는 경우 빈 리스트 ([ ])가 나온다. 

슬라이싱의 예시를 이용하여, 실제 어떻게 동작하는지 알아보자 

````{toggle} 슬라이싱 예제 보기 

```python
arr = [5, 7, 9, 0, 3, 1, 6, 2, 4, 8]

print(arr[:3]) # 인덱스 3 이전까지 오른쪽으로 1씩 이동하여 슬라이싱, 결과 [5, 7, 9]
print(arr[-3:]) # -3부터 끝까지 오른쪽으로 1씩 이동하여 슬라이싱, 결과 [2, 4, 8]
print(arr[:-3]) # 0부터 인덱스 -3 이전까지 오른쪽으로 1씩 이동하여 슬라이싱, 결과 [5, 7, 9, 0, 3, 1, 6]
print(arr[:-3:-1]) # 0부터 -3이전까지 왼쪽으로 1씩 이동하여 슬라이싱, 결과 [8, 4]
print(arr[:-3:-2]) # 0부터 -3이전까지 왼쪽으로 2씩 이동하여 슬라이싱, 결과 [8]
print(arr[:-5:-2]) # 0부터 -3이전까지 왼쪽으로 2씩 이동하여 슬라이싱, 결과 [8 ,2]
print(arr[:3:-1]) # 0부터 3이전까지 왼쪽으로 1씩 이동하여 슬라이싱, [8, 4, 2, 6, 1, 3]
print(arr[0:3:-1]) # step의 방향과 start end의 방향이 다름. 
```
````
`````
## 선택 정렬 (Selection Sort)

선택 정렬은 매 스텝마다 가장 작은 것을 선택해 앞으로 보내는 과정을 반복 수행하는 알고리즘이다. 즉, 데이터가 무작위로 있을 때, '아직 정렬되지 않은 범위 속 가장 작은 데이터'를 선택해 '아직 정렬되지 않은 범위의 맨 앞에 있는 데이터'와 바꾸고, 그 다음 작은 데이터를 선택해 앞에서 두 번째 데이터와 바꾸는 과정을 반복한다. 이렇게 "데이터"를 "선택"하여 바꾸는 작업을 통해 정렬하는 알고리즘을 (데이터) 선택 정렬이라고 한다. 

데이터의 개수를 N=10이라고 할때, 다음 예제를 확인해보자. 

### 선택 정렬 예시 

앞으로 그림에서, 회색 카드는 '현재 정렬되지 않은 데이터 중에서 가장 작은 데이터'를 의미하며, 하늘색 카드는 '이미 정렬된 데이터'를 의미한다. 

**Step 1**
![1](../../assets/img/sort/1.png)

초기 단계에서는 모든 데이터가 정렬되어 있지 않으므로, 전체 중에서 가장 작은 데이터를 선택한다. 
'7': current element to be swapped (the first element in the remaining unordered section)
'0': smallest element in the remaining unsorted section 

위의 데이터 '0'과 '7'를 바꾼다. 

**Step 2**

![2](../../assets/img/sort/2.png)

데이터가 정렬되어 있지 않은 범위 중 가장 작은 데이터를 선택하여 현재 원소와 바꾼다. 

'5': current element to be swapped (the first element in the remaining unordered section)
'1': smallest element in the remaining unsorted section 


**Step 3**

![3](../../assets/img/sort/3.png)

데이터가 정렬되어 있지 않은 범위 중 가장 작은 데이터를 선택하여 현재 원소와 바꾼다. 

'9': current element to be swapped (the first element in the remaining unordered section)
'2': smallest element in the remaining unsorted section 

**Step 4**

![4](../../assets/img/sort/4.png)

데이터가 정렬되어 있지 않은 범위 중 가장 작은 데이터를 선택하여 현재 원소와 바꾼다. 

'7': current element to be swapped (the first element in the remaining unordered section)
'3': smallest element in the remaining unsorted section 

**Step 5**

![5](../../assets/img/sort/5.png)

데이터가 정렬되어 있지 않은 범위 중 가장 작은 데이터를 선택하여 현재 원소와 바꾼다. 

'7': current element to be swapped (the first element in the remaining unordered section)
'4': smallest element in the remaining unsorted section 


**Step 6**

![6](../../assets/img/sort/6.png)

데이터가 정렬되어 있지 않은 범위 중 가장 작은 데이터를 선택하여 현재 원소와 바꾼다. 

'5': current element to be swapped (the first element in the remaining unordered section)
'5': smallest element in the remaining unsorted section 


**Step 7**

![7](../../assets/img/sort/7.png)

데이터가 정렬되어 있지 않은 범위 중 가장 작은 데이터를 선택하여 현재 원소와 바꾼다. 

'6': current element to be swapped (the first element in the remaining unordered section)
'6': smallest element in the remaining unsorted section 

**Step 8**

![8](../../assets/img/sort/8.png)

데이터가 정렬되어 있지 않은 범위 중 가장 작은 데이터를 선택하여 현재 원소와 바꾼다. 

'9': current element to be swapped (the first element in the remaining unordered section)
'7': smallest element in the remaining unsorted section 

**Step 9**

![9](../../assets/img/sort/9.png)

데이터가 정렬되어 있지 않은 범위 중 가장 작은 데이터를 선택하여 현재 원소와 바꾼다. 

'9': current element to be swapped (the first element in the remaining unordered section)
'8': smallest element in the remaining unsorted section 

**Step 10**

![10](../../assets/img/sort/10.png)

데이터가 정렬되어 있지 않은 범위 중 가장 작은 데이터를 선택하여 현재 원소와 바꾼다. 

'9': current element to be swapped (the first element in the remaining unordered section)
'9': smallest element in the remaining unsorted section 

### 선택 정렬 코드 및 시간 복잡도

````{admonition} selection sort source code 
:class: dropdown 

```{code-block} python
---
caption: selection sort python source code 
---

arr = [7, 5, 9, 0, 3, 1, 6, 2, 4, 8]

for i in range(len(arr)):
  min_index = i 
  for j in range(i+1, len(arr)):
    if arr[min_index] > arr[j]:
      min_index = j 
  
  # SWAP
  arr[i], arr[min_index] = arr[min_index], arr[i] 

print(arr)
```
````

위의 구현 방식으로 연산 횟수는 (N) + (N-1) + (N-2) + ... + 2 로 볼 수 있다. 따라서 N(N+1)/2로 연산을 수행한다고 가능하면 이는 간단히 O($N^2$)로 표현할 수 있다. 
이러한 선택 정렬은 기본 정렬 라이브러리를 포함하여 뒤에서 다룰 알고리즘과 비교했을 때 매우 비효율적이다. 다만, 특정한 리스트에서 가장 적은 데이터를 찾는 일이 코딩 테스트에서 잦으므로 선택 정렬 소스코드 형태에 익숙해질 필요가 있다. 자주 작성하여서 선택 정렬 소스코드에 익숙해지자. 

## 삽입 정렬 (Insertion Sort)

삽입 정렬은 데이터를 하나씩 확인하여, 각 데이터를 삽입하기에 적절한 "위치"는 어디일까? 라는 접근 방법을 가진 알고리즘이다. 삽입 정렬은 필요할 때만 위치를 바꾸므로 '데이터가 거의 정렬되어 있을 때' 훨씬 효율적이다. 위에서 살펴본 것처럼 선택 정렬은 현재 데이터의 상태와 상관없이 무조건 모든 원소를 비교하고 위치를 바꾸지만 삽입 정렬은 아니다. 

삽입 정렬 특징 
- 특정한 데이터가 적절한 위치에 들어가기 이전에, 그 앞까지의 데이터는 이미 정렬되어 있다고 가정 
- 선택 정렬처럼 언제나 모든 원소를 비교하고 위치를 바꾸지 않는다. 


  
### 삽입 정렬 예시 

**Step 1** 

![11](../../assets/img/sort/11.png)

삽입 정렬은 두 번째 데이터부터 시작한다. 왜냐하면 첫번째 데이터는 그 자체로 정렬되어 있다고 판단하기 때문이다. 

**Step 2**
![12](../../assets/img/sort/12.png)

첫 번째 데이터 '7'은 그 자체로 정렬되어 있다고 판단하고, 두 번째 데이터인 '5'가 어떤 위치가 들어갈지 판단한다. '7'의 왼쪽으로 들어가거나 (기존 '7'의 위치) 혹은 오른쪽으로 들어가는 두 경우 (현재 '5'의 위치)만 존재한다. 우리는 카드를 오름차순으로 정렬하고자 하므로 '7'의 왼쪽에 삽입한다.

** 그림에서 인덱스는 현재 수가 있는 자리의 왼쪽 화살표로 표시됨. 

**Step 3**
![13](../../assets/img/sort/13.png)
'9'가 들어갈 위치를 판단해야한다. 삽입될 수 있는 총 위치는 3가지이며 파란색 화살표로 9가 들어갈 위치를 표시하지만, 원래 자리이기 때문에 그대로둔다. 


**Step 4**
![14](../../assets/img/sort/14.png)
'0'이 들어갈 위치를 판단해야한다. 파란색 화살표 자리 즉, 현재 '5'가 있는 위치에 삽입한다. 

**Step 5**
![15](../../assets/img/sort/15.png)
'3'이 들어갈 위치를 판단해야한다. 파란색 화살표 자리 즉, 현재 '5'가 있는 위치에 삽입한다. 

**Step 6**
![16](../../assets/img/sort/16.png)
'1'이 들어갈 위치를 판단해야한다. 파란색 화살표 자리 즉, 현재 '3'이 있는 위치에 삽입한다. 

**중략**
중간 과정을 생략한다. 

**Step 7**
![17](../../assets/img/sort/17.png)
'4'가 들어갈 위치를 판단한다. '5'의 자리에 삽입한다. 

**Step 8**
![18](../../assets/img/sort/18.png)
'8'이 들어갈 위치를 판단한다. '9'의 자리에 삽입한다. 

**Step 9**
![19](../../assets/img/sort/19.png)
이와 같이 적절한 위치에 삽입하는 과정을 N-1번 반복하게 되면 위와 같이 모든 데이터가 정렬된 것을 확인할 수 있다. 

### 삽입 정렬 코드 및 시간 복잡도 

삽입 정렬에서, 정렬이 이루어져 있는 원소들 (파란색)은 항상 오름차순을 유지하고 있기 땜누에, 현재 특정한 데이터가 삽입될 위치를 선정할때, (삽입될 위치를 찾기 위하여 왼쪽으로 한칸씩 이동할때) 삽입될 데이터보다 작은 데이터를 만나면 그 위치에서 멈추면 된다. 

````{admonition} code for insertion sort 
:class: dropdown

```python
arr = [7, 5, 9, 0, 3, 1, 6, 2, 4, 8]

for i in range(1, len(arr)):
    for j in range(i, 0, -1): # 인덱스 i부터 1까지 감소하며 반복하는 문법 
        if arr[j] < arr[j-1]: # 한 칸씩 왼쪽으로 이동 
            arr[j], arr[j-1] = arr[j-1], arr[j]
        else: # 자기보다 작은 데이터를 만나면 그 위치에서 멈춤 
            break 

print(arr)
```
````

위의 코드에서 드는 질문은, 왜 list.insert(index, value) 함수를 쓰지 않는가?이다. 즉, 최종 위치를 찾아서 그곳에 현재 원소를 넣으면 안되는가? 아래 두 가지 이유를 들 수 있다. 

(1) 시간 복잡도 면에서 비슷하거나 더 느림
- insert()를 쓰면 내부적으로 리스트의 모든 원소를 한 칸씩 뒤로 미는 연산(O(N)) 이 발생합니다.
- 그런데 어차피 삽입 정렬도 O(N²)이기 때문에, 굳이 insert()로 더 많은 내부 처리를 할 필요가 없어요.

(2) 교환(swap) 기반 구현이 구조적으로 더 명확함
- 삽입 정렬은 배열을 직접 순회하면서 인접 원소를 비교하고 교환하는 구조로 되어 있어요.
- 이 방식이 언어나 자료구조가 달라도 동일하게 동작하는 일반적인 알고리즘 형태라서, 교육·면접용 코드에서도 swap으로 표현합니다.

삽입 정렬의 시간 복잡도는 O(N^2)으로, 선택 정렬과 마찬가지로 반복문이 2번 중첩되어 있다. 하지만 꼭 기억해야할 점은 선택 정렬과 다르게, 현재 리스트의 데이터가 거의 정렬 되어 있는 상태라면 최선의 경우 O(N)의 시간 복잡도를 가질 수 있다. 다음에 배울 퀵 정렬과 비교했을 때 보통은 삽입 정렬이 비효율적이나 정렬이 거의 되어 있는 상황에서는 퀵 정렬 알고리즘보다 더 강력하다. 


````{admonition} Algorithm summary of insertion sort 
:class: important 

삽입 정렬 알고리즘 정리
- 리스트의 두 번째 원소부터 시작해서, 그 앞쪽 부분(이미 정렬된 부분) 에서 자신이 들어갈 적절한 위치를 찾은 뒤 삽입하는 방식
- 즉, arr[0:i] 구간은 항상 정렬된 상태를 유지하면서 arr[i]를 그 안에 알맞게 “넣는” 과정을 반복합니다.
- 시간 복잡도: O($N^2$)
````

## 퀵 정렬 (Quick Sort)

퀵 정렬은 정렬 알고리즘 중에서 가장 많이 사용되는 알고리즘이다. ***퀵 정렬은 기준을 설정한 다음 큰 수와 작은 수를 교환한 후 리스트를 반으로 나누는 방식***으로 동작한다. 이해하는데 오래걸리지만, 원리를 이해하면 병합 정렬, 힙 정럴 등 다른 고급 정렬 기법에 비해 쉽게 소스코드를 작성할 수 있다. 

피벗(Pivot)이라는 개념이 등장하는데, 큰 숫자와 작은 숫자를 교환할 때, 교환하기 위한 '기준'을 바로 피벗이라고 표현한다. 퀵 정렬을 수행하기 전에는 피벗을 어떻게 설정할 것인지 미리 명시해야 한다. 사실 퀵 정렬은 피벗 설정, 리스트 분할 방법에 따라 여러 가지 방식으로 나뉘는데, 오늘 수업에서는 가장 대표적인 분할 방식인 **호어 분할(Hoare Partition) 방식을 기준으로 퀵 정렬을 설명하겠다. 

**호어 분할 방식**
- 피벗 설정: 리스트에서 첫 번째 데이터를 피벗으로 정한다. 
- 정렬 방식: 왼쪽에서부터 피벗보다 큰 데이터를 찾고, 오른쪽에서부터 피벗보다 작은 데이터를 찾는다. 그 다음 큰 데이터와 작은 데이터의 위치를 서로 교환해준다. 이 과정을 반복하면 '피벗'에 대하여 정렬이 수행된다. 

다음과 같이 초기 데이터가 구성되어있다고 가정해보자. 해당 알고리즘을 파트 1, 2, 3로 나누어서 살펴보겠다. 

![20](../../assets/img/sort/20.png)

### 파트 1 

**Step 1**
![21](../../assets/img/sort/21.png)

피벗: 리스트의 첫 번째 데이터인 '5' 
피벗의 왼쪽: 왼쪽에서부터 '5'보다 큰 데이터 선택 '7' 
피벗의 오른쪽: 오른쪽에서부터 '5'보다 작은 데이터 선택 '4' 
swap: 위에서 선택된 '7'과 '4'의 위치를 서로 변경 

**Step 2**
![22](../../assets/img/sort/22.png)

피벗: 리스트의 첫 번째 데이터인 '5' 
피벗의 왼쪽: 아까 바꿨던 위치에서부터 '5'보다 큰 데이터 선택 '9' 
피벗의 오른쪽: 아까 바꿨던 위치에서부터'5'보다 작은 데이터 선택 '2' 
swap: 위에서 선택된 '9'과 '2'의 위치를 서로 변경 

**Step 3**
![23](../../assets/img/sort/23.png)

피벗: 리스트의 첫 번째 데이터인 '5' 
피벗의 왼쪽: 아까 바꿨던 위치에서부터 '5'보다 큰 데이터 선택 '6' 
피벗의 오른쪽: 아까 바꿨던 위치에서부터'5'보다 작은 데이터 선택 '1' 
-> 이 때 현재 왼쪽에서부터 찾는 값과 오른쪽에서부터 찾는 값의 위치가 서로 엇갈린 것을 알 수 있다. 이렇게 두 값의 위치가 엇갈린 경우에는 '작은 데이터'와 '피벗'의 위치를 서로 변경한다. 
swap: 위에서 선택된 '9'과 '2'의 위치를 서로 변경 

**Step 4: 분할 완료**
![24](../../assets/img/sort/24.png)

이와 같이 '피벗'이 이동한 상태에 도달하면 분할 완료이다. 분할 완료 후에는 위의 그림처럼 피벗 데이터의 왼쪽 데이터는 모두 피벗 값인 '5'보다 작고, 오른쪽에 있는 데이터는 모두 '5'보다 크다. 

즉, 분할 (Divide) 또는 파티션 (Partition)이란, 피벗의 왼쪽에는 피벗보다 작은 데이터가 위치하고, 피벗의 오른쪽에는 피벗보다 큰 데이터가 위치하도록 하는 작업을 일컫는다. 

### 파트 2: 분할 완료된 피벗의 왼쪽 리스트

분할 완료된 피벗의 왼쪽 리스트에서는 다음과 같이 정렬이 진행되며 구체적인 정렬과정은 동일하다. 
![25](../../assets/img/sort/25.png)

### 파트 3: 분할 완료된 피벗의 오른쪽 리스트 

분할 완료된 피벗의 오른쪽 리스트에서는 다음과 같이 정렬이 진행되며 구체적인 정렬과정은 동일하다. 아래 그림에서는 첫 단계에서, 오른쪽과 왼쪽의 데이터가 서로 엇갈리는 지점에는 작은 데이터와 피벗의 데이터가 동일하다. 따라서, swap하고 난 후 분할 완료된 리스트는 첫 리스트와 동일함을 알 수 있다. 
![26](../../assets/img/sort/26.png)

특정한 리스트에서 피벗을 설정하여 정렬을 수행한 이후, 피벗을 기준으로 왼쪽 리스트와 오른쪽 리스트에서 각각 다시 정렬 수행한다. 재귀 함수와 동작 원리가 같고, 따라서, 종료 조건이 필요하다. 퀵 정렬이 끝나는 조건은 ***리스트의 개수가 1개***인 경우이다. 리스트의 원소가 1개라면, 이미 정렬되어 있다고 간주할 수 있으며 분할이 불가능하다. 

### 퀵 정렬 전체 정리 과정 그림 
![27](../../assets/img/sort/27.png)

### 퀵 정렬 소스코드 및 시간 복잡도 

````{admonition} source code for Quick sort 
:class: dropdown 

```{code-block} python
arr = [5, 7, 9, 0, 3, 1, 6, 2, 4, 8]

def quick_sort(array, start, end):
    if start >= end:  # Stop when the subarray has one or zero elements
        return

    pivot = start
    left = start + 1
    right = end

    while True:
        # Move left pointer until you find an element greater than the pivot
        while left <= end and array[left] <= array[pivot]:
            left += 1
        # Move right pointer until you find an element smaller than the pivot
        while right > start and array[right] >= array[pivot]:
            right -= 1

        if left > right:
            array[right], array[pivot] = array[pivot], array[right]
            break
        else:
            array[left], array[right] = array[right], array[left]

    # After partitioning, recursively sort the left and right subarrays
    quick_sort(array, start, right - 1)
    quick_sort(array, right + 1, end)

quick_sort(arr, 0, len(arr) - 1)
print(arr)

```
````

퀵 정렬의 평균 시간 복잡도는 O(NlogN)이다. 앞선 두 정렬 알고리즘에 비해 매우 빠른 편이다. 간단히 이런 시간 복잡도를 가지는 이유에 대해서 설명해보자면, 데이터의 개수가 8개라고 가정하고 다음과 같이 정확히 절반씩 나눈다고 도식화를 해보자. 이때 '높이'를 확인해보면, 데이터의 개수가 N개일 때 높이는 약 logN이라고 판단할 수 있다. Here, the “height” of the recursion tree refers to the depth of recursive partitioning, not the number of swaps or partitions performed.

즉, 
1. 한 번의 분할(partition) 과정에서 모든 원소(N개) 를 한 번씩 비교하므로,한 단계(partition depth)당 O(N) 의 시간이 걸립니다.
2. 이상적인 경우(매번 절반씩 나뉜다면), 전체 트리의 높이는 log₂N 단계가 됩니다.
3. 따라서 총 수행 시간은 O(N) * O(log N) = O(N log N) 이 됩니다.

즉, 아래 그림에서 “높이 = 재귀 깊이(분할 단계 수)”이고, 각 단계마다 N개의 비교가 일어나는 구조입니다.

![28](../../assets/img/sort/28.png)

퀵 정렬의 재귀는 두 방향으로 뻗지만, 각 단계에서 처리되는 총 원소 수의 합은 항상 N이므로, 전체 시간은 O(N)(단계당) × O(log N)(단계 수) = O(N log N) 이 된다.


일반적으로 컴퓨터 과학에서 log의 의미는 밑이 2인 로그를 의미한다. 즉, $log_{2}N$을 의미하며 데이터의 개수 1000일 때, $log_{2}N$는 10 정도이다. 즉, N=1000일 때 10은 상대적으로 매우 작은 수이다. 데이터의 개수가 많을 수록 차이는 매우 극명하게 드러난다. 

## 계수 정렬 

## 파이썬의 정렬 라이브러리 
## 정리 

````{admonition} Summary 
- 선택 정렬: swap할 "데이터"를 선택 
- 삽입 정렬: Swap할 "인덱스"를 선택 
- 퀵 정럴: 특정한 리스트에서 피벗을 설정하여 정렬을 수행한 이후, 피벗을 기준으로 왼쪽 리스트와 오른쪽 리스트에서 각각 다시 정렬 수행. 

````
