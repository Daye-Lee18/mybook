# Variable Scope 

변수의 스코프(Scope)는 ***변수가 유효한 범위*** 를 의미한다. 스코프 개념을 이해하면, 특히 재귀 함수나 DFS처럼 함수를 반복적으로 호출하는 문제에서 예상치 못한 변수 꼬임(Bug)을 방지할 수 있다.

먼저 변수 스코프에 대해 공부하기 전에 keyword (예약어)가 무엇인지 공부한 후 변수 스코프를 어떻게 사용할 수 있는지 알아보자. 

## Keyword (예약어)

keyword란 ***파이썬 문법에서 특별한 역할을 하는 예약된 단어*** 이다. 이러한 keyword는 문법적으로 정해진 기능을 수행하므로 변수 이름으로 사용할 수 없다. 

**종류**
- if, for, while, def, return, class, global, nonlocal, True, None ... 

```python
import keyword 
print(keyword.kwlist)
```

출력 예시: 
```bash
['False', 'None', 'True', 'and', 'as', 'assert', 'break', 
 'class', 'continue', 'def', 'del', 'elif', 'else', 'except',
 'finally', 'for', 'from', 'global', 'if', 'import', 
 'in', 'is', 'lambda', 'nonlocal', 'not', 'or', 'pass', 
 'raise', 'return', 'try', 'while', 'with', 'yield']
```

## 변수 스코프 (scope) 

파이썬은 변수를 찾을 때 아래 순서로 탐색한다. 

LEGB 규칙 
- Local $\rightarrow$ Enclosing $\rightarrow$ Global $\rightarrow$ Built-in


```{list-table} LEGB 규칙 정리 
:widths: 15 45 45
:header-rows: 1 
:name: LEGB 규칙

* - Scope
  - Explanation
  - Example 
* - Local
  - 현재 함수 내부에서 정의된 변수 
  - def func(): x = 1 
* - Enclosing 
  - 중첩 함수 (outer function)의 변수 
  - def outer(): x = 1; def inner(): print(x)
* - Global 
  - 모듈 (파일) 전체에서 접근 가능한 변수
  - 파일 상단에 정의된 변수 
* - Built-in
  - 파이썬이 기본 제공하는 이름 
  - len, sum, print 등 
```

## global, nonlocal 키워드 

### global 
특징 
- 전역 변수임을 명시하여, 함수 안에서 전역 변수를 수정할 때 사용한다. 
- global 키워드를 선언하지 않고 전역 변수와 같은 이름의 변수를 만들면, 함수 내부에 새로 지역 변수가 만들어지므로 전역 변수에 영향을 주지 않는다. 
- 전역 변수는 함수 안에서 처음 생성 가능하다. 즉, 전역 스코프에 아직 존재하지 않아도, global x로 선언하고 값을 할당하면 함수 실행 시점에 새 전역 변수가 생성됨. 

따라서, 위의 세번째 특징에서는 함수가 호출될때 전역 변수가 생성되므로, 함수 호출 전 해당 변수에 접근하면 오류가 난다. 

```python
def create_var():
    global x
    x = 99

print(x)    # ❌ NameError: name 'x' is not defined
create_var()
print(x)    # ✅ 99
```

```{list-table} global 정리
:widths: 15 45 45 
:header-rows: 1

* - 상황
  - 결과
  - 설명
* - 함수 내부에서 `global x` 선언 + 할당 
  - 전역 변수 새로 생성 
  - 함수 실행 시 전역 스코프에 등록됨 
* - 전역에 이미 `x`가 있음. 
  - 기존 전역 변수 수정 
  - 새로 만들지 않고 기존 것 덮어 씀 
* - `global`없이 지역 변수로 할당 
  - 전역 변수로 등록되지 않음 
  - 함수 바깥에서 접근 불가 
* - 함수 실행 전 전역 접근 
  - NameError 
  - 전역에 아직 생성되지 않음 
```


### nonlocal 
특징 
- 중첩 함수(inner function)안에서 ***바로 한 단계 바깥 함수의 변수*** 를 수정하고 싶을 때 사용 
- global과 마찬가지로, 해당 keyword없이 outer function과 동일한 변수이름을 가진 변수를 생성하는 경우, 새로운 지역 변수가 만들어져서, outer function의 변수에는 영향을 미치지 않는다. 

```{list-table} global vs. nonlocal 비교표 
:widths: 15 45 45 
:header-rows: 1

* - 구분 
  - global 
  - nonlocal 
* - 접근 대상 
  - 모듈 전역 변수 
  - 한 단계 바깥 함수의 지역 변수 
* - 사용 위치 
  - 함수 내부 
  - 중첩 함수 (inner function) 내부 
* - 새 변수 생성 여부 
  - 전역 변수 재사용 
  - 상위 함수의 변수 재사용 
* - 사용할 수 없는 곳 
  - 중첩 함수가 없는 곳에선 의미 없음 (전역 스코프에서 사용하면 의미 없음, 전역 변수임이 자명하기 때문)
  - global 스코프에는 사용 불가
* - 대표 예시 
  - 전역 설정, 카운터 등 
  - 클로저, 내부 함수 상태 유지 

```

## mutable 객체 vs. immutable 객체 

| 구분 | mutable | immutable |
|---|---|---|
| 정의 | 내부 데이터를 변경할 수 있음| 내부 데이터를 변경할 수 없음 |
| 대표 타입 | list, dict, set | int, float, str, tuple |
| 개체가 함수 인자로 넘겨지면 (전달 방식) | 참조(reference) 전달 | 참조(reference)전달 ***(단, 값 변경 불가)*** | 
| 함수 내부 수정 시 | 원본 까지 변경됨 | 원본은 변경되지 않음 |

두 객체 모두 reassign은 가능하다. 하지만, 내부 데이터를 변경할 수 있느냐 없느냐에 따라 mutable과 immutable이 나뉜다. 
list와 같은 mutable 객체가 함수 인자로 넘겨졌을 때, 재할당 (reassign)과 수정 (modify)의 차이를 이해해야한다. 변경 가능한 객체는 내용을 바꾸면 원본도 변하지만, "다른 객체로" 할당하면 원본은 그대로가 된다. 

파이썬은 “call by reference”나 “call by value”보다 call by object reference (call by sharing, 객체의 참조를 전달) 방식을 사용한다. 즉, 인자로 전달된 객체를 직접 수정하면 원본이 바뀌지만, 새 객체로 재할당하면 원본과의 연결이 끊어진다. 단지 그 객체가 immutable이기 때문에 "수정이 불가능해서", 마치 값이 복사된 것처럼 보인다. 

````{admonition} 코드 예시
:class: dropdown 

```{code-block} python 
def change_string(s):
    print("before:", id(s))
    s += " world"          # 문자열은 불변, 새 객체 생성
    print("after:", id(s))
    return s

msg = "hello"
print("msg id:", id(msg))
change_string(msg)
print("msg id after:", id(msg))
```

출력: 
```bash
msg id: 139962626660528
before: 139962626660528
after: 139962626661744  ← 새로운 문자열 객체 생성
msg id after: 139962626660528

```
s가 처음엔 msg와 같은 객체를 참조했지만, " world"를 더하는 순간 새 str 객체가 만들어졌다. 따라서 원본 msg는 영향을 받지 않음. 
````

즉, call by object reference 는 "객체의 참조"가 전달되어 mutable이면 내부 변경 가능하고 immutable이면 새 객체로 대체된다. 

```{list-table} 객체 vs. 참조 vs. 주소 
:widths: 15 50 
:header-rows: 1 

* - 용어 
  - 의미 
* - 객체 (object)
  - 실제 데이터가 저장된 메모리 블록 
* - 참조 (reference)
  - 그 객체를 가리키는 변수의 내부 정보 (주소를 내부적으로 보관)
* - 주소 (address)
  - 메모리 상에서 객체가 존재하는 실제 위치 (`id()` 값)
```
즉, 참조는 “주소 자체”가 아니라, “주소를 통해 객체를 가리키는 추상적 이름표”라고 생각하는 게 가장 정확합니다.

<img src="../../assets/img/scope/1.png" widths="500px">

```python
a = [1, 2, 3]
b = a

print(a is b)     # ✅ True, 같은 객체를 참조
print(id(a))      # 예: 140487543893312
print(id(b))      # 같은 주소
```

```python
# 함수 인자와 연결 
def modify(x):
    print("before:", id(x))
    x.append(4)
    print("after :", id(x))

lst = [1, 2, 3]
modify(lst)
```
```bash
before: 140460077558528
after : 140460077558528
```
- 함수 인자로 전달된 x는 lst와 같은 객체를 참조 
- x.append()는 같은 객체의 내용을 바꿈 
- 즉, call by sharing (call by object reference)가 작동하고 있음 
  
코딩 테스트 문제를 풀 때, 특히 dfs와 같이 재귀적으로 함수를 계속 부를 때 이러한 변수들을 잘못 쓰면 꼬여서, 문제를 틀릴 가능성이 높다. 

**리스트를 다룰 때** 
````{admonition} 예시 1
:class: dropdown 

내용 수정은 원본 반영, 재할당은 연결이 끊김.
- arr과 nums는 처음에는 같은 리스트를 가리킴
- 하지만 arr = [100, 200] 문장은
    - “새로운 리스트를 생성하고 arr 변수가 그것을 가리키도록 바꾸는 것”
    - 원래의 nums와의 연결이 끊어진다. 
    - 따라서, 원본에 영향을 주지 않음. 
```{code-block} python
def modify_list(arr):
    arr.append(4)     # ✅ 원본이 변경됨
    arr = [100, 200]  # ❌ 새로운 리스트를 가리켜도 원본엔 영향 없음

nums = [1, 2, 3]
modify_list(nums)
print(nums)  # ✅ [1, 2, 3, 4]

```
````

````{admonition} 예시 2
:class: dropdown 

복사본 만들기, 슬라이싱 [:]
```{code-block} python
def dfs(path):
    path.append(1)     # 원본 변경
    new_path = path[:] # ✅ 안전한 복사본
    new_path.append(2)
    print("path =", path)
    print("new_path =", new_path)

dfs([])

# path = [1]
# new_path = [1, 2]


```
````

````{admonition} 예시 3
:class: dropdown 

DFS나 재귀 함수에서 전역 리스트를 조절할 때, ***인자로 넘기지 않고*** 함수 바깥에서 직접 접근해야 하는 경우도 있습니다.

```{code-block} python
res = []

def dfs(node):
    global res      # 전역 리스트 사용
    if not node:
        return
    res.append(node.val)
    dfs(node.left)
    dfs(node.right)
```

아래처럼 클로저(closure) 내부에서 값 누적이 필요할 때, 
```{code-block} python
def outer():
    path = []
    def dfs(node):
        nonlocal path
        if node:
            path.append(node)
            dfs(node.left)
            dfs(node.right)
    dfs(root)

```
````

````{admonition} 정리
:class: important 

- global, nonlocal은 스코프 제어를 위한 keyword.
- list와 같은 mutable 객체는 “수정”과 “재할당”의 차이를 구분해야 한다.
- 재귀 함수 안에서 리스트나 카운터를 잘못 공유하면 값이 꼬여서 엉뚱한 답이 나올 수 있다.
- 매 재귀 호출마다 새 리스트를 써야 할 때는 반드시 path[:]로 복사.
- 전역적 누적이 필요한 값은 global이나 nonlocal로 제어.
- DFS, 백트래킹, 조합 생성 문제에서는 이 개념이 특히 중요하다.
````

## 코테 실수 예시 

틀린 DFS 코드 

```python
def dfs(node, path, result):
    if not node:
        result.append(path)     # ❌ path가 전부 같은 리스트 객체를 참조
        return
    path.append(node.val)
    dfs(node.left, path, result)
    dfs(node.right, path, result)

res = []
dfs(root, [], res)
print(res)  # 모든 경로가 동일하게 출력됨
```

올바른 DFS 코드 
```python
def dfs(node, path, result):
    if not node:
        result.append(path[:])  # ✅ 복사본을 저장
        return
    path.append(node.val)
    dfs(node.left, path, result)
    dfs(node.right, path, result)
    path.pop()                  # ✅ 백트래킹 시 원상복구

```

## 실습 문제 

````{admonition} 전역 변수 생성 시점 확인 
:class: dropdown 

```{code-block} python
def create_var():
    global count
    count = 1

# 아래 중 어느 줄에서 에러가 날까?
print(count)
create_var()
print(count)
```

답: 
- 첫번째 print(count)에서 NameError 
- create_var() 호출 후에는 1 출력 
````

````{admonition} 리스트 참조 꼬임 확인 
:class: dropdown 

```{code-block} python
def test(arr):
    arr.append(1)
    arr = [99]
    return arr

nums = []
test(nums)
print(nums)  # ?

```

예상 출력: 
- [1]
- 새 리스트 [99]는 함수 안에서만 존재
````


````{admonition} nonlocal 실습 
:class: dropdown 

```{code-block} python
def outer():
    val = 0
    def inner():
        nonlocal val
        val += 1
        print(val)
    inner()
    inner()

outer()
```

예상 출력: <br> 
1<br>
2<br>
````
