# Final (2026 1st semester — Data Science) — Solutions

## Short answer problems

- For each question below, fill in the blanks labeled `(1)`, `(2)`, `(3)`, …, or briefly write the requested result.
- Record all short-answer responses on your own answer sheet, clearly numbered to match the questions.
- Each correctly completed blank is worth **2 points** unless otherwise noted.
- A one-page cheat sheet is allowed. Calculators are not.
- Assume Python 3 syntax.

### Q1. Data analysis pipeline

Fill in each blank.

- **(1)** Data science emphasizes ___ . *(Choose one word: `correlation` or `causation`.)*
- **(2)** Observing that two variables move together does **not** prove ___ . *(Choose one word: `correlation` or `causation`.)*

The standard 5-step analysis pipeline taught in class is, in order:

> **(3)**  →  strategy planning  →  data collection  →  **(4)**  →  **(5)**

Each of (3), (4), (5) is one of the following exact terms — use each **exactly once**, in the order it appears in the pipeline:

`analysis`, `application`, `problem definition`

- **(6)** "Districts with more CCTVs have more reported crimes" is the classic CCTV example. True or False: this proves that adding CCTVs *causes* more crime.

```{admonition} A1
:class: dropdown

(1) correlation <br>
(2) causation <br>
(3) problem definition <br>
(4) analysis <br>
(5) application <br>
(6) False (it only shows correlation, not causation) <br>
```

### Q2. NumPy broadcasting

```python
import numpy as np
a = np.array([[1, 2, 3], [4, 5, 6]])   # shape (2, 3)
b = np.array([10, 20, 30])             # shape (3,)
c = np.array([[100], [200]])           # shape (2, 1)
```

- **(1)** The shape of `a + b` is `___`. *(Write as a Python tuple, e.g. `(2, 3)`.)*
- **(2)** The first row of `a + b` is `___`. *(Write as a Python list.)*
- **(3)** The shape of `a + c` is `___`.
- **(4)** The second row of `a + c` is `___`.
- **(5)** Broadcasting compares trailing dimensions. Two dimensions are compatible when they are equal **or** when one of them equals ___ . *(One number.)*
- **(6)** `a + np.array([1, 2])` raises a `___Error`. *(Fill in the error-class name.)*

```{admonition} A2
:class: dropdown

(1) (2, 3) <br>
(2) [11, 22, 33] <br>
(3) (2, 3) <br>
(4) [204, 205, 206] <br>
(5) 1 <br>
(6) Value (a ValueError) <br>
```

### Q3. NumPy `axis` argument

```python
import numpy as np
arr = np.array([[1, 2, 3],
                [4, 5, 6]])
print(arr.sum())
print(arr.sum(axis=0))
print(arr.sum(axis=1))
```

- **(1)** `arr.sum()` returns `___` *(one integer)*.
- **(2)** `arr.sum(axis=0)` returns `___` *(list)*.
- **(3)** `arr.sum(axis=0)` has shape `___` *(tuple)*.
- **(4)** `arr.sum(axis=1)` returns `___` *(list)*.
- **(5)** `arr.sum(axis=1)` has shape `___` *(tuple)*.
- **(6)** For a 2-D array of shape `(H, W)`, `axis=0` collapses the dimension whose length is `___`. *(Write `H` or `W`.)*

```{admonition} A3
:class: dropdown

(1) 21 <br>
(2) [5, 7, 9] <br>
(3) (3,) <br>
(4) [6, 15] <br>
(5) (2,) <br>
(6) H <br>
```

### Q4. Boolean masking and `np.where`

```python
import numpy as np
a = np.array([10, 25, 7, 33, 18, 5])

print(a[a > 15])              # (1)
print(np.where(a > 15))       # (2)
print(np.where(a > 15, a, 0)) # (3)
```

- **(1)** The first output is `___` *(write the resulting array, e.g. `[…]`)*.
- **(2)** The second output is `(array([___]),)`. Fill the list of integers inside the array.
- **(3)** Called with a single argument, `np.where(cond)` returns the ___ where the condition is True. *(One word: `values` or `indices`.)*
- **(4)** The third output is `___` *(write the resulting array)*.
- **(5)** `np.where(cond, x, y)` returns `x` when `cond` is ___ and `y` otherwise. *(Write `True` or `False`.)*

```{admonition} A4
:class: dropdown

(1) [25 33 18] <br>
(2) 1, 3, 4 <br>
(3) indices <br>
(4) [ 0 25  0 33 18  0] <br>
(5) True <br>
```

### Q5. Pandas boolean indexing — `.loc` vs `.iloc`

```python
import pandas as pd
df = pd.DataFrame({
    'name':  ['Anna', 'Ben', 'Cora', 'Dan'],
    'score': [85, 60, 92, 70]
})
# df.index defaults to 0, 1, 2, 3
```

- **(1)** To select all rows where `score >= 80`, write `df[ ___ ]`. *(Fill in a single boolean expression.)*
- **(2)** Using `.loc`, to select only the `name` column of those rows: `df.loc[ ___ , 'name']`. *(Same boolean expression as (1).)*
- **(3)** `df.loc[1:3]` returns the rows with labels `___` *(write a comma-separated list)*.
- **(4)** `df.iloc[1:3]` returns the rows at positions `___` *(write a comma-separated list)*.
- **(5)** Therefore `.loc` slicing is ___ of the end label, while `.iloc` slicing is ___ of the end position. *(Each blank: `inclusive` or `exclusive`.)*

```{admonition} A5
:class: dropdown

(1) df['score'] >= 80 <br>
(2) df['score'] >= 80 <br>
(3) 1, 2, 3 <br>
(4) 1, 2 <br>
(5) inclusive ; exclusive <br>
```

### Q6. Pandas `pivot_table`

```python
import pandas as pd
df = pd.DataFrame({
    'class':  ['A', 'A', 'B', 'B', 'A', 'B'],
    'gender': ['M', 'F', 'M', 'F', 'M', 'F'],
    'score':  [80, 90, 70, 100, 60, 95]
})

pt = df.pivot_table(index='class', columns='gender', values='score', aggfunc='mean')
```

- **(1)** The row labels of `pt` are the unique values of column `'___'`.
- **(2)** The column labels of `pt` are the unique values of column `'___'`.
- **(3)** Each cell holds the ___ of `score`. *(One word, matching `aggfunc`.)*
- **(4)** The numerical value at row `'A'`, column `'M'` is `___`.
- **(5)** The numerical value at row `'B'`, column `'F'` is `___`.

```{admonition} A6
:class: dropdown

(1) class <br>
(2) gender <br>
(3) mean <br>
(4) 70   (mean of 80 and 60) <br>
(5) 97.5 (mean of 100 and 95) <br>
```

### Q7. IQR-based outlier detection

Consider the sorted data `data = [3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 30]`. Use the linear-interpolation definition of quantiles (the default in `numpy.quantile` and `pandas.Series.quantile`).

- **(1)** Q1 (25th percentile) = `___`
- **(2)** Q3 (75th percentile) = `___`
- **(3)** IQR (= Q3 − Q1) = `___`
- **(4)** Lower fence (= Q1 − 1.5·IQR) = `___`
- **(5)** Upper fence (= Q3 + 1.5·IQR) = `___`
- **(6)** List every value in `data` that the IQR rule flags as an outlier (a value below the lower fence or above the upper fence): `___`
- **(7)** The Z-score outlier method assumes the data follows a `___` distribution. *(One word.)*

```{admonition} A7
:class: dropdown

(1) 6.5    (position 2.5: arr[2]=6 → arr[3]=7, midpoint) <br>
(2) 11.5   (position 7.5: arr[7]=11 → arr[8]=12, midpoint) <br>
(3) 5 <br>
(4) −1 <br>
(5) 19 <br>
(6) 30 <br>
(7) normal (Gaussian) <br>
```

### Q8. Long vs. wide format

Suppose each of **4 students** records 3 subject scores: `math`, `eng`, `sci`.

- **(1)** In **wide** format, the table has one row per student, and `math`, `eng`, `sci` each appear as their own ___ of the table. *(One word: `row` or `column`.)*
- **(2)** In wide format the table has `___` row(s) of data. *(One integer.)*
- **(3)** In **long** format, the same data is stored with one row per (student, subject) pair. The long table has `___` row(s) of data. *(One integer.)*
- **(4)** The pandas method that converts a **wide** table into a **long** table is `df.___(...)`. *(Method name only.)*
- **(5)** The pandas method that converts a **long** table back into a **wide** table is `df.___(...)`. *(Method name only — write the most common one.)*

```{admonition} A8
:class: dropdown

(1) column <br>
(2) 4 <br>
(3) 12   (4 students × 3 subjects) <br>
(4) melt <br>
(5) pivot   (also accept: pivot_table) <br>
```

### Q9. Choosing the right chart

For each scenario, write the single most appropriate chart from this list:

`barplot`, `countplot`, `histplot`, `boxplot`, `lineplot`, `scatterplot`, `heatmap`, `pairplot`

- **(1)** Show the frequency of each category in a single categorical column.
- **(2)** Show the distribution of a single continuous variable using bins.
- **(3)** Show the median, quartiles, and outliers of a continuous variable, separately for each group.
- **(4)** Show the relationship between two continuous variables.
- **(5)** Show a correlation matrix of many features with each cell annotated.
- **(6)** Show how a value changes over time.
- **(7)** Show pairwise relationships across **all pairs** of numerical columns at once.

```{admonition} A9
:class: dropdown

(1) countplot <br>
(2) histplot <br>
(3) boxplot <br>
(4) scatterplot <br>
(5) heatmap <br>
(6) lineplot <br>
(7) pairplot <br>
```

### Q10. Matplotlib `fig` vs `axes` and the Axes API

`plt.subplots(...)` returns a tuple `(fig, ax)`.

- **(1)** `fig` is the ___ object — the entire canvas (window) that contains the plot(s). *(One word, capitalized.)*
- **(2)** `ax` (also written `axes`) is the ___ object — each rectangular plotting region where data is drawn. One figure can hold many of them. *(One word, capitalized.)*
- **(3)** After `plt.subplots(1, 2)`, `ax` is a 1-D array of length `___`. *(One integer.)*
- **(4)** After `plt.subplots(2, 3)`, `ax` is a 2-D array of shape `___`. *(Write as a Python tuple.)*
- **(5)** With the result of `plt.subplots(2, 3)`, the **bottom-right** subplot is accessed as `ax[___]`. *(Write the bracketed index expression using non-negative integers, e.g. `[a][b]`.)*
- **(6)** True or False: `plt.subplots()` called **with no row/column arguments** also returns `ax` as an array. `___`
- **(7)** The Axes-level method that sets the title of one specific subplot is `ax.___("...")`. *(Method name only.)*

Fill in the blanks of the following code, which produces a 1 × 2 figure: histogram on the left, scatter on the right.

```python
import matplotlib.pyplot as plt
import seaborn as sns

fig, ax = plt.subplots(__(8)__, __(9)__, figsize=(10, 4))
sns.histplot(data=df, x='score', ax=__(10)__)
sns.scatterplot(data=df, x='study_hours', y='score', ax=__(11)__)
ax[0].__(7)__("Score distribution")
ax[1].__(7)__("Score vs Study hours")
plt.show()
```

- **(12)** To color points in a seaborn scatterplot by a categorical column such as `gender`, pass the keyword argument `___='gender'`. *(Argument name only.)*

```{admonition} A10
:class: dropdown

(1) Figure <br>
(2) Axes <br>
(3) 2 <br>
(4) (2, 3) <br>
(5) [1][2] <br>
(6) False (it returns a single Axes object, not an array) <br>
(7) set_title <br>
(8) 1 <br>
(9) 2 <br>
(10) ax[0] <br>
(11) ax[1] <br>
(12) hue <br>
```

### Q11. Predicting the resulting plot

For each code snippet, fill in the blanks. Whenever asked for a "chart type," write the **seaborn function name only** (lowercase, one word), chosen from:
`barplot`, `countplot`, `histplot`, `boxplot`, `lineplot`, `scatterplot`, `heatmap`.

**(a)**
```python
sns.countplot(data=df, x='gender')
```
- **(1)** Chart type (seaborn function name): `___`
- **(2)** The number of bars in the plot equals the number of unique values of column `'___'`.
- **(3)** The height of each bar represents the ___ of rows in that category. *(One word, matching what `countplot` literally computes.)*

**(b)**
```python
sns.histplot(data=df, x='score', bins=20, kde=True)
```
- **(4)** Chart type (seaborn function name): `___`
- **(5)** The x-axis is split into `___` equal-width bins. *(One integer.)*
- **(6)** The smooth curve overlaid on top of the bars is called the `___` curve. *(3-letter abbreviation.)*

**(c)**
```python
sns.scatterplot(data=df, x='study_hours', y='score', hue='gender')
```
- **(7)** Chart type (seaborn function name): `___`
- **(8)** The `hue='gender'` argument maps the `gender` column to each point's ___ . *(One word: which visual property — `position`, `color`, `size`, or `shape`?)*

**(d)**
```python
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
sns.boxplot(data=df, x='class', y='score', ax=ax[0])
sns.lineplot(data=df, x='year', y='sales', ax=ax[1])
```
- **(9)** Number of rows in the subplot grid: `___` *(integer)*
- **(10)** Number of columns in the subplot grid: `___` *(integer)*
- **(11)** Left subplot — chart type (seaborn function name): `___`
- **(12)** Right subplot — chart type (seaborn function name): `___`
- **(13)** The right subplot's x-axis shows the column `'___'`.

```{admonition} A11
:class: dropdown

(1) countplot <br>
(2) gender <br>
(3) count <br>
(4) histplot <br>
(5) 20 <br>
(6) KDE <br>
(7) scatterplot <br>
(8) color <br>
(9) 1 <br>
(10) 2 <br>
(11) boxplot <br>
(12) lineplot <br>
(13) year <br>
```

---

## Problem solving problems

### PS1. Outlier removal, group aggregation, and heatmap

Fill in the blanks so that the program:

1. Replaces extreme values in `'score'` with `NaN` using the **IQR method** (Q1 = 25th percentile, Q3 = 75th percentile, fences at Q1 − 1.5·IQR and Q3 + 1.5·IQR).
2. Computes the mean `score` per (`class`, `gender`).
3. Reshapes the result into a wide pivot table.
4. Plots it as an annotated heatmap.

```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.DataFrame({
    'class':  ['A','A','A','B','B','B','C','C','C'],
    'gender': ['M','F','M','F','M','F','M','F','M'],
    'score':  [85, 92, 78, 65, 999, 70, 88, 91, -50]
})

# Step 1 — IQR outlier filter on 'score'
q1   = df['score'].quantile(__(1)__)        # 25th percentile (one decimal number)
q3   = df['score'].quantile(__(2)__)        # 75th percentile (one decimal number)
iqr  = __(3)__                              # one expression in q1, q3
low  = q1 - 1.5 * iqr
high = q3 + 1.5 * iqr
df.loc[(df['score'] < low) | (df['score'] > high), 'score'] = __(4)__   # missing-value marker

# Step 2 & 3 — pivot table of mean score by class × gender
pt = df.pivot_table(
        index=__(5)__,                      # column name as a string
        columns=__(6)__,                    # column name as a string
        values='score',
        aggfunc=__(7)__)                    # aggregation name as a string

# Step 4 — annotated heatmap
sns.__(8)__(pt, annot=True, cmap='Blues')    # seaborn function name
plt.title("Mean score by class and gender")
plt.show()
```

````{admonition} PS1 solution
:class: dropdown

```python
q1   = df['score'].quantile(0.25)
q3   = df['score'].quantile(0.75)
iqr  = q3 - q1
low  = q1 - 1.5 * iqr
high = q3 + 1.5 * iqr
df.loc[(df['score'] < low) | (df['score'] > high), 'score'] = np.nan

pt = df.pivot_table(
        index='class',
        columns='gender',
        values='score',
        aggfunc='mean')

sns.heatmap(pt, annot=True, cmap='Blues')
plt.title("Mean score by class and gender")
plt.show()
```

Blanks:
(1) 0.25 <br>
(2) 0.75 <br>
(3) q3 - q1 <br>
(4) np.nan <br>
(5) 'class' <br>
(6) 'gender' <br>
(7) 'mean'   (also accept: np.mean) <br>
(8) heatmap <br>

After Step 1, both `999` and `-50` become `NaN`, so `pivot_table` (which ignores NaN) computes the mean of the remaining valid values.
````

### PS2. Clean, aggregate, and visualize

Fill in the blanks so that the program:

1. Drops every row that contains at least one `NaN`.
2. Adds a new column `'total' = math + eng`.
3. Computes the mean `total` per `gender`.
4. Plots the result as a bar chart using the matplotlib **Axes** API, with a title and labeled axes.

```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.DataFrame({
    'name':   ['철수', '영희', '민수', '지수', '재현', '소영'],
    'gender': ['M',   'F',   'M',   'F',   'M',   'F'],
    'math':   [90,    85,    np.nan, 70,   60,    95],
    'eng':    [80,    90,    70,    np.nan, 65,   88]
})

# Step 1 — drop rows containing any NaN
clean = df.__(1)__()                         # method name only

# Step 2 — add a 'total' column
clean['total'] = clean[__(2)__] + clean[__(3)__]   # column names as strings

# Step 3 — mean 'total' per gender
avg = clean.__(4)__('gender')['total'].__(5)__()    # (4) groupby method,  (5) aggregation method
print(avg)

# Step 4 — barplot using the Axes API
fig, ax = plt.subplots(__(6)__, __(7)__, figsize=(6, 4))   # (6) rows, (7) cols
sns.__(8)__(x=avg.index, y=avg.values, ax=ax)              # seaborn function name
ax.__(9)__("Average Total by Gender")                       # Axes method that sets title
ax.__(10)__("Gender")                                       # Axes method that sets x-axis label
ax.__(11)__("Average total score")                          # Axes method that sets y-axis label
plt.show()
```

Answer the following short questions about the same code.

- **(12)** After Step 1, the number of rows in `clean` is `___`. *(One integer.)*
- **(13)** In the resulting bar chart, the **(13)**-axis shows the average total score. *(Write `x` or `y`.)*

````{admonition} PS2 solution
:class: dropdown

```python
clean = df.dropna()
clean['total'] = clean['math'] + clean['eng']

avg = clean.groupby('gender')['total'].mean()
print(avg)

fig, ax = plt.subplots(1, 1, figsize=(6, 4))
sns.barplot(x=avg.index, y=avg.values, ax=ax)
ax.set_title("Average Total by Gender")
ax.set_xlabel("Gender")
ax.set_ylabel("Average total score")
plt.show()
```

Blanks:
(1) dropna <br>
(2) 'math' <br>
(3) 'eng' <br>
(4) groupby <br>
(5) mean <br>
(6) 1 <br>
(7) 1 <br>
(8) barplot <br>
(9) set_title <br>
(10) set_xlabel <br>
(11) set_ylabel <br>
(12) 4   (철수, 영희, 재현, 소영 remain; 민수 has NaN math, 지수 has NaN eng) <br>
(13) y <br>

Numerical sanity check:
- After dropna: 철수 total = 170,  영희 = 175,  재현 = 125,  소영 = 183.
- Mean by gender — M: (170 + 125) / 2 = 147.5;  F: (175 + 183) / 2 = 179.
````
