# 입력 

이진 탐색 문제처럼 입력 데이터가 많거나, 탐색 범위가 매우 넓은 편의 경우 input()함수를 사용하면 오히려 동작 속도가 느려서 시간  초과로 오답 판정을 받을 수 있다. 이처럼 입력 데이터가 많은 문제는 sys 라이브러리의 readline()함수를 이용하면 된다. 

```python
import sys 

input = sys.stdin.readline
```

sys 라이브러리를 사용할때는 입력 후 엔터<font size='2'>Enter</font>가 줄 바꿈 기호로 입력되는데, 이 공백 문자를 제거하려면 rstrip()함수를 사용해야한다. 

또한, 다른 .txt 파일에 저장된 테스트 케이스를 불러오고 싶은 경우 다음 코드를 이용하면 된다. 

```python
import sys 

sys.stdin = open('input.txt')

n, m = map(int, input().split())
```
