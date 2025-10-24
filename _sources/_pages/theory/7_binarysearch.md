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

# Lecture 7-1. Binary Search 

## 순차 탐색 (Sequential Search)

***순차 탐색 (Sequential Search)***이란, 리스트 안에 있는 특정한 데이터를 찾기 위해 앞에서부터 데이터를 하나씩 차례대로 확인하는 방법이다. 보통 정렬되지 않은 리스트에서 데이터를 찾아야 할 때 사용한다. 리스트 내에 데이터가 아무리 많아도 시간만 충분하다면 항상 원하는 원소 (데이터)를 찾을 수 있다는 장점이 있다. 

````{admonition} source code for sequential search
:class: dropdown 

```{code-block} python
---
caption: 순차 탐색으로 Dongbin을 찾는 탐색 과정. 소스 코드를 실행하면 정상적으로 이름 문자열이 몇 번째 데이터인지 출력하는 것을 알 수 있다. 
---

def sequential_search(n, target, array):
    for i in range(n):
        if array[i] == target:
            return i + 1 # 현재의 위치 반환 (인덱스는 0부터 시작하므로 1 더하기)
        
print("생성할 원소 개수를 입력한 다음 한 칸 띄고 찾을 문자열을 입력하세요.")
input_data = input().split()
n = int(input_data[0])
target = input_data[1]

print("앞서 적은 원소 개수만큼 문자열을 입력세요. 구분은 띄어쓰기 한 칸으로 합니다.")
array = input().split()

print(sequential_search(n, target, array))
```
````
순차 탐색의 특징은 ***데이터의 정렬 여부와 상관없이*** 가장 앞에 있는 원소부터 하나씩 확인해야한다는 것이다. 데이터의 개수가 N개 일 때 최대 N번의 비교 연산이 필요하므로 순차 탐색은 최악의 경우 시간 복잡도 O(N)이다. 

## 이진 탐색 (Binary Search)

이진 탐색은 배열 내부의 **데이터가 정렬되어 있어야만** 사용할 수 있는 알고리즘이다. 

## 트리 자료 구조 

## 이진 탐색 트리 