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

# 문제 풀이 순서 팁 (Tips)

0. (사전) 코딩테스트 IDE 준비 
    - 필수. 빠른 입출력 세팅 (sys.stdin.readline 등), 자주 쓰이는 함수 (예: dfs/bfs, 이분탐색) 미리 준비해두면 좋습니다. 
    - 다만, 온라인 IDE 환경과 로컬 IDE가 다르므로 실제 환경에 맞게 미리 연습해두는 게 핵심입니다. 

1. 복잡도 파악 
    - 문제를 읽고 **제약 조건(입력 크기, 시간 제한)** 먼저 확인 $\rightarrow$ 바로 시간복잡도 상한을 설정해야 합니다. 
    - **풀이 아이디어 $\rightarrow$ 복잡도 $\rightarrow$ 자료구조** 선택을 메모하는 습관 추천
 
2. 에지 케이스 (edge case) 정리 
   - 중요 단계! 
     - 최소 입력값 (N=0, N=2)
     - 최대 입력값 (시간복잡도 한계에 닿는 값)
     - 특수 입력 (중복, 음수, 동일 값 등)
     - 출력 애매한 케이스 (답이 여러 개 나올 수 있는 경우)

3. 코드 짜기 및 디버깅
   - 기본 구조 먼저 (입력 처리 $\rightarrow$ 로직 함수화 $\rightarrow$ 출력)
   - 디버깅하기 편하게 **단계별로 프린트** 넣었다가 제출 직전 삭제 
     - 디버깅하면서 그때그때 바뀌는 것을 확인하는 것도 좋지만, 
     - 별도의 file (touch rotation.py)을 만들어서 print로 변경 전, 변경 후 원하는 결과와 같은지 확인해보는 것이 빠르다. 
     - **출력 형식 오류 (띄어쓰기, 개행)** 체크 필수. 
   - 특히 구현 문제는 `작은 함수 단위`로 쪼개는 게 실수 줄이는 데 도움 됩니다. 

4. 디버깅 
   1. 시간 초과 시: 입력 방식, 자료구조, 알고리즘 복잡도 재확인 
   2. 틀렸습니다 시: 정리해둔 에지 케이스를 넣어서 로컬에서 재현해보기 