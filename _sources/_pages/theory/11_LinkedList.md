# Lecture 11-1. Linked List 

Lecture 7-1에서 배웠듯, Tree는 class Tree를 이용하여 pointer로 연결하여 build_tree, tree_to_list 함수와 같은 helper 함수를 이용해 ***테스트 케이스*** 를 활용해 답안의 정답 유무를 빠르게 확인할 수 있었다. 테스트 케이스를 통해 정답 유무를 보고 싶은 경우 위의 helper 함수를 이용하는 것이 편리하지만, 그래프 자체를 만들어서 관리하거나 더 빠른 시간안에 검색을 하고 싶은 경우에는 해당 방법은 그리좋지 않다. 따라서, edges 리스트를 받으면 parent와 children 리스트로 변환해서 반환하는 함수를 배웠었다. 

이와 같은 작업은 Linked list 관련 문제를 푸는데에도 활용될 수 있다. 

