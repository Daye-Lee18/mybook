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

# Lecture 5-1. Shortest Path

최단 경로 (Shotest Path) 알고리즘은 가장 짧은 경로를 찾는 알고리즘이다. 그래서 "길 찾기" 문제라고도 불린다. 최단 경로 알고리즘 유형에는 다양한 종류가 있다. 예를 들어 '한 지점에서 다른 특정 지점까지의 최단 경로를 구해야 하는 경우", "모든 지점에서 다른 모든 지점까지의 최단 경로를 모두 구해야하는 경우" 등의 다양한 사례가 존재한다. 각 사례에 맞는 알고리즘을 알고 있다면 좀 더 쉽게 풀 수 있다. 이번 챕터에서는 **다익스트라 최단 경로**와 **플로이드 워셔 알고리즘** 유형을 다룰 것이다. 

사실 이번 장에서 다루는 내용은 그리디 알고리즘 및 다이나믹 프로그래밍 알고리즘의 한 유형으로 볼 수 있다. 

## Dijkstra Algorithm 



