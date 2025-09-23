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

# Lecture 5-1. 정렬 

`정렬 (Sorting)`이란 데이터를 특정한 기준에 따라서 순서대로 나열하는 것을 말한다. 정렬 알고리즘으로 데이터를 정렬하면 이후에 배울 `이진 탐색 (Binary Search)`이 가능해진다. 이번 수업에서는 총 4가지 정렬 (선택 정렬, 삽입 정렬, 퀵 정렬, 계수 정렬)에 대해서 배운다. 이 장에서 다루는 예제는 모두 `오름차순 (Increasing order)` 정렬을 수행한다고 가정한다. 내림차순 정렬은 리스트의 원소를 뒤집는 (Reverse) 메서드를 제공하기 때문에 그 방법을 사용하여 O(N) 시간 복잡도로 만들 수 있다.

## 선택 정렬 (Selection Sort)
