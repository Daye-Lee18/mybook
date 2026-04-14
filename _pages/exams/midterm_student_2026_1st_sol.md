# Midterm

## Short answer problems

- For each question below, fill in the blanks labeled `(1)`, `(2)`, `(3)`, …, or briefly write the requested result.
- Record all short-answer responses on your own answer sheet, clearly numbered to match the questions.
- Each correctly completed blank is worth 2 points unless otherwise noted.
- Assume Python 3 syntax.

### Q1. List indexing and nested list access

Consider the following code.

```python
data = ['A', ['B', 'C', ['D', 'E']], 'F']
```

Fill in the blanks so that the code prints `E`.

```python
print(data[(1)][(2)][(3)])
```

:::{.callout-note collapse="true" title="A1"}
A1. <br>
(1) 1 <br>
(2) 2 <br>
(3) 1 <br>
:::

### Q2. List slicing and direction

Consider the following code.

```python
arr = [10, 20, 30, 40, 50, 60, 70]
print(arr[5:1:-2])
print(arr[1:5:-1])
```

Fill in the blanks.

- The first print result is (1).
- The second print result is (2), because (3) and (4) do not match.

:::{.callout-note collapse="true" title="A2"}
A2. <br>
(1) [60, 40] <br>
(2) [] <br>
(3) start/end direction <br>
(4) step direction <br>
:::

### Q3. List methods and return values

Consider the following code.

```python
nums = [3, 1, 2]
a = nums.sort()
b = sorted(nums, reverse=True)
```

Fill in the blanks.

- After the code runs, `nums` is (1).
- `a` is (2).
- `b` is (3).
- The reason `a` is not a sorted list is that `list.sort()` sorts the list (4).

:::{.callout-note collapse="true" title="A3"}
A3. <br>
(1) [1, 2, 3] <br>
(2) None <br>
(3) [3, 2, 1] <br>
(4) in-place <br>
:::

### Q4. Shallow copy of list

Consider the following code.

```python
origin = [1, 2, [3, 4]]
copied = origin.copy()
copied[2][0] = 100
```

Fill in the blanks.

- After execution, `origin` becomes (1).
- This happens because `copy()` performs a (2) copy, so the nested list is still (3).

:::{.callout-note collapse="true" title="A4"}
A4. <br>
(1) [1, 2, [100, 4]] <br>
(2) shallow <br>
(3) shared / referenced by both lists <br>
:::

### Q5. Dictionary access and safe lookup

Consider the following code.

```python
scores = {'철수': 90, '영희': 85}
```

Fill in the blanks.

- `scores['민수']` causes a (1).
- To avoid an error and receive `None` when the key does not exist, use (2).
- If you want to check existence first, use the operator (3).

:::{.callout-note collapse="true" title="A5"}
A5. <br>
(1) KeyError <br>
(2) scores.get('민수') <br>
(3) in <br>
:::

### Q6. Dictionary update vs insertion

Consider the following code.

```python
fruit = {'apple': 2, 'banana': 3}
fruit['banana'] = 10
fruit['grape'] = 4
```

Fill in the blanks.

- After execution, `fruit['banana']` is (1).
- Key `'grape'` is (2).
- The statement `dict[key] = value` does (3) if the key exists, and does (4) if the key does not exist.

:::{.callout-note collapse="true" title="A6"}
A6. <br>
(1) 10 <br>
(2) added / inserted <br>
(3) update / modify <br>
(4) insertion / addition <br>
:::

### Q7. Tuple vs list

State True or False.

(1) A tuple can be used to store values that should not be modified.  
(2) A tuple supports `append()` just like a list.  
(3) To make a one-element tuple, a trailing comma is needed.  
(4) A tuple is immutable.

:::{.callout-note collapse="true" title="A7"}
A7. <br>
(1) True <br>
(2) False <br>
(3) True <br>
(4) True <br>
:::

### Q8. Set properties

Consider the following code.

```python
s = {'apple', 'banana', 'apple', 'kiwi'}
```

Fill in the blanks.

- The number of elements in `s` is (1).
- A set removes (2) values automatically.
- A set does not support indexing because it has no (3).

:::{.callout-note collapse="true" title="A8"}
A8. <br>
(1) 3 <br>
(2) duplicate <br>
(3) order / index <br>
:::

### Q9. Default parameter and multiple return values

Consider the following code.

```python
def calc(x, y=5):
    total = x + y
    return total, total * 2

a, b = calc(3)
c = calc(4, 6)
```

Fill in the blanks.

- After `a, b = calc(3)`, `a` is (1) and `b` is (2).
- `c` is of type (3), with value (4).
- `y=5` in the function definition is a (5), which is used when no corresponding argument is provided by the caller.

:::{.callout-note collapse="true" title="A9"}
A9. <br>
(1) 8 <br>
(2) 16 <br>
(3) tuple <br>
(4) (10, 20) <br>
(5) default parameter (기본값 매개변수) <br>
:::

### Q10. Return and early exit

Consider the following code.

```python
def f(x):
    if x > 0:
        return x * 2
    print('A')
    return -1
    print('B')
```

Fill in the blanks.

- `f(3)` returns (1).
- `f(0)` prints (2) and returns (3).
- The line `print('B')` is never executed because `return` makes the function (4).

:::{.callout-note collapse="true" title="A10"}
A10. <br>
(1) 6 <br>
(2) A <br>
(3) -1 <br>
(4) exit / terminate immediately <br>
:::

### Q11. Packing with asterisk

Consider the following code.

```python
def mydata(*values):
    return values

result = mydata(10, 20, 30)
```

Fill in the blanks.

- `result` is stored as a (1) type.
- Its value is (2).
- In a function parameter, `*` plays the role of (3) multiple arguments into one tuple.

:::{.callout-note collapse="true" title="A11"}
A11. <br>
(1) tuple <br>
(2) (10, 20, 30) <br>
(3) packing <br>
:::

### Q12. Variable scope

Consider the following code.

```python
balance = 100

def deposit():
    balance = 50
    print(balance)

deposit()
print(balance)
```

Fill in the blanks.

- The first printed value is (1).
- The second printed value is (2).
- The `balance` inside `deposit()` is a (3) variable.
- The `balance` outside the function is a (4) variable.

:::{.callout-note collapse="true" title="A12"}
A12. <br>
(1) 50 <br>
(2) 100 <br>
(3) local <br>
(4) global <br>
:::

### Q13. Missing value handling

Consider the following code.

```python
import pandas as pd

df = pd.DataFrame({
    '이름': ['철수', '영희', '민수'],
    '수학': [85, None, 72],
    '영어': [90, 88, None]
})
```

Fill in the blanks.

- `df.isnull().sum().sum()` returns (1).
- To remove all rows that contain at least one NaN value, use (2).
- To replace all NaN values in `df['수학']` with `0`, use (3).
- To sort `df` by `수학` in descending order, use (4).

:::{.callout-note collapse="true" title="A13"}
A13. <br>
(1) 2 <br>
(2) df.dropna() <br>
(3) df['수학'] = df['수학'].fillna(0) <br>
(4) df.sort_values(by='수학', ascending=False) <br>
:::

### Q14. Numpy array vs list

State True or False.

(1) A numpy array usually stores elements of the same data type.  
(2) For `arr = np.array([1,2,3])`, `arr * 2` repeats the array like a Python list.  
(3) Vectorized operations are one reason numpy is faster than Python lists for large numeric computation.  
(4) `['a', 'b'] * 2` and `np.array([1,2]) * 2` behave in exactly the same way.

:::{.callout-note collapse="true" title="A14"}
A14. <br>
(1) True <br>
(2) False <br>
(3) True <br>
(4) False <br>
:::

### Q15. Numpy indexing and slicing

Consider the following code.

```python
import numpy as np

ar = np.arange(12).reshape(3, 4)
print(ar[1:][1:])
print(ar[1:, 1:])
```

Fill in the blanks.

- The first result has shape (1).
- The second result has shape (2).
- These results are different because `ar[1:][1:]` applies slicing (3), while `ar[1:, 1:]` slices the original array by (4) and (5) at the same time.

:::{.callout-note collapse="true" title="A15"}
A15. <br>
(1) (1, 4) <br>
(2) (2, 3) <br>
(3) sequentially / in two steps <br>
(4) rows <br>
(5) columns <br>
:::

### Q16. Pandas Series slicing

Consider the following code.

```python
import pandas as pd

s = pd.Series([10, 20, 30, 40], index=['a', 'b', 'c', 'd'])
print(s['b':'c'])
```

Fill in the blanks.

- Label-based slicing in a Series includes the (1) label.
- Therefore the printed values are (2) and (3).

:::{.callout-note collapse="true" title="A16"}
A16. <br>
(1) end / last <br>
(2) 20 <br>
(3) 30 <br>
:::

### Q17. DataFrame selection: `iloc` vs `loc`

Consider the following code.

```python
import pandas as pd

df = pd.DataFrame(
    [[90, 80], [70, 60], [50, 40]],
    index=['철수', '영희', '민수'],
    columns=['수학', '영어']
)
```

Fill in the blanks.

- `df.iloc[1, 0]` returns (1).
- `df.loc['영희', '수학']` returns (2).
- `iloc` uses (3)-based indexing, while `loc` uses (4)-based indexing.

:::{.callout-note collapse="true" title="A17"}
A17. <br>
(1) 70 <br>
(2) 70 <br>
(3) position <br>
(4) label <br>
:::

### Q18. Missing values after DataFrame addition

Consider the following code.

```python
import pandas as pd

d1 = pd.DataFrame({'국어':[80, 90]}, index=['철수', '영희'])
d2 = pd.DataFrame({'국어':[70, 60]}, index=['영희', '민수'])
d3 = d1 + d2
```

Fill in the blanks.

- In `d3`, the row for `'영희'` becomes (1).
- The row for `'철수'` becomes (2).
- The row for `'민수'` becomes (3).
- This is because pandas aligns data by matching (4) and (5) before calculation.

:::{.callout-note collapse="true" title="A18"}
A18. <br>
(1) 160 <br>
(2) NaN <br>
(3) NaN <br>
(4) index <br>
(5) column labels <br>
:::

### Q19. Groupby and aggregation

Consider the following code.

```python
import pandas as pd

df = pd.DataFrame({
    '학급': ['A', 'A', 'B', 'B', 'A'],
    '수학': [90, 80, 70, 100, 60]
})
```

Fill in the blanks.

- `df.groupby('학급')['수학'].mean()` computes the (1) of `수학` for each (2).
- For class `A`, the result is (3).
- For class `B`, the result is (4).

:::{.callout-note collapse="true" title="A19"}
A19. <br>
(1) mean / average <br>
(2) group / class <br>
(3) 230/3 ≈ 76.67 <br>
(4) 85 <br>
:::

### Q20. Datetime conversion and extraction

Consider the following code.

```python
import pandas as pd

df = pd.DataFrame({'생년월일':['1990-03-02', '1991-08-08']})
df['생년월일'] = pd.to_datetime(df['생년월일'])
df['년'] = df['생년월일'].dt.year
df['월'] = df['생년월일'].dt.month
```

Fill in the blanks.

- `pd.to_datetime()` converts text data into (1) type.
- The values in column `년` are (2).
- The values in column `월` are (3).
- The `.dt` accessor is used to extract (4) information from datetime data.

:::{.callout-note collapse="true" title="A20"}
A20. <br>
(1) datetime <br>
(2) 1990, 1991 <br>
(3) 3, 8 <br>
(4) date / time component <br>
:::
