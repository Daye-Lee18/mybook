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

## DP 문제 예시 

DFS를 배울 때, 피보나치 수열을 구현하는 방법을 배웠다. 피보나치 수열은 다음과 같은 형태로 끝없이 이어진다. 