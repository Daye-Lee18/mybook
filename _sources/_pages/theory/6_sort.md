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

삽입 정렬은 데이터를 하나씩 확인하여, 각 데이터를 삽입하기에 적절한 위치는 어디일까? 라는 접근 방법을 가진 알고리즘이다. 삽입 정렬은 필요할 때만 위치를 바꾸므로 '데이터가 거의 정렬되어 있을 때' 훨씬 효율적이다. 위에서 살펴본 것처럼 선택 정렬은 현재 데이터의 상태와 상관없이 무조건 모든 원소를 비교하고 위치를 바꾸지만 삽입 정렬은 아니다. 

### 삽입 정렬 예시 
