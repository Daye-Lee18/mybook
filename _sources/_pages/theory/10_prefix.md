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

# prefix_sum[i]: data[0] ~ data[i]까지의 합
prefix_sum = [0] * len(data)
prefix_sum[0] = data[0]

for i in range(1, len(data)):
    prefix_sum[i] = prefix_sum[i-1] + data[i]

print("Prefix Sum:", prefix_sum)
```
출력: 
```bash
Prefix Sum: [1, 3, 6, 10, 15]
```
