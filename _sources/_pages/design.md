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

# 수업 디자인 컴포넌트 모음

본 페이지는 강의 자료 제작에 유용한 **디자인 컴포넌트 샘플**입니다.  
카드/그리드, 토글, 탭, 알림 상자, 퀴즈, 코드 복사, Colab 배지 등을 한곳에 모았습니다.

---

## 2. 알림 상자 (admonition)

```{admonition} 실습 전 준비
:class: tip
구글 계정 로그인 후 각 장의 **Open in Colab** 버튼을 눌러 노트북을 엽니다.
```

## 3. 연습문제

**문제 1.** 사용자로부터 정수 `n`을 입력받아 1부터 `n`까지의 합을 출력하시오.

```{toggle}
**해설 / 정답 코드**
```python
n = int(input())
print(sum(range(1, n+1)))
