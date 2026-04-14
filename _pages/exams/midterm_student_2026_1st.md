# Midterm (2026 1st semester python basic)

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


### Q5. Dictionary access and safe lookup

Consider the following code.

```python
scores = {'철수': 90, '영희': 85}
```

Fill in the blanks.

- `scores['민수']` causes a (1).
- To avoid an error and receive `None` when the key does not exist, use (2).
- If you want to check existence first, use the operator (3).


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


### Q7. Tuple vs list

State True or False.

(1) A tuple can be used to store values that should not be modified.  
(2) A tuple supports `append()` just like a list.  
(3) To make a one-element tuple, a trailing comma is needed.  
(4) A tuple is immutable.


### Q8. Set properties

Consider the following code.

```python
s = {'apple', 'banana', 'apple', 'kiwi'}
```

Fill in the blanks.

- The number of elements in `s` is (1).
- A set removes (2) values automatically.
- A set does not support indexing because it has no (3).


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


### Q14. Numpy array vs list

State True or False.

(1) A numpy array usually stores elements of the same data type.  
(2) For `arr = np.array([1,2,3])`, `arr * 2` repeats the array like a Python list.  
(3) Vectorized operations are one reason numpy is faster than Python lists for large numeric computation.  
(4) `['a', 'b'] * 2` and `np.array([1,2]) * 2` behave in exactly the same way.


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

