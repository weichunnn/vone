[Skip to main content](#main-content)

Back to top  

 Ctrl+K

[Home](/api)

cudf

[cucim](/api/cucim/stable)[cudf-java](/api/cudf-java/stable)[cudf](/api/cudf/stable/)[cugraph](/api/cugraph/stable)[cuml](/api/cuml/stable)[cuproj](/api/cuproj/stable)[cuspatial](/api/cuspatial/stable)[cuvs](/api/cuvs/stable)[cuxfilter](/api/cuxfilter/stable)[dask-cuda](/api/dask-cuda/stable)[dask-cudf](/api/dask-cudf/stable)[kvikio](/api/kvikio/stable)[libcudf](/api/libcudf/stable/namespacecudf/)[libcuml](/api/libcuml/stable)[libcuproj](/api/libcuproj/stable)[libcuspatial](/api/libcuspatial/stable)[libkvikio](/api/libkvikio/stable)[libucxx](/api/libucxx/stable)[raft](/api/raft/stable)[rapids-cmake](/api/rapids-cmake/stable)[rmm](/api/rmm/stable)

stable (24.10)

[nightly (24.12)](/api/cudf/nightly/)[stable (24.10)](/api/cudf/stable/)[legacy (24.08)](/api/cudf/legacy/)

*   [GitHub](https://github.com/rapidsai/cudf "GitHub")
*   [Twitter](https://twitter.com/rapidsai "Twitter")

# 10 Minutes to cuDF and Dask cuDF[#](#minutes-to-cudf-and-dask-cudf "Permalink to this heading")

Modelled after 10 Minutes to Pandas, this is a short introduction to cuDF and Dask cuDF, geared mainly towards new users.

## What are these Libraries?[#](#what-are-these-libraries "Permalink to this heading")

[cuDF](https://github.com/rapidsai/cudf) is a Python GPU DataFrame library (built on the Apache Arrow columnar memory format) for loading, joining, aggregating, filtering, and otherwise manipulating tabular data using a DataFrame style API in the style of [pandas](https://pandas.pydata.org).

[Dask](https://dask.org/) is a flexible library for parallel computing in Python that makes scaling out your workflow smooth and simple. On the CPU, Dask uses Pandas to execute operations in parallel on DataFrame partitions.

[Dask cuDF](https://github.com/rapidsai/cudf/tree/main/python/dask_cudf) extends Dask where necessary to allow its DataFrame partitions to be processed using cuDF GPU DataFrames instead of Pandas DataFrames. For instance, when you call `dask_cudf.read_csv(...)`, your cluster’s GPUs do the work of parsing the CSV file(s) by calling [`cudf.read_csv()`](https://docs.rapids.ai/api/cudf/stable/api_docs/api/cudf.read_csv.html).

**Note:** This notebook uses the explicit Dask cuDF API (dask\_cudf) for clarity. However, we strongly recommend that you use Dask's [configuration infrastructure](https://docs.dask.org/en/latest/configuration.html) to set the "dataframe.backend" option to "cudf", and work with the Dask DataFrame API directly. Please see the [Dask cuDF documentation](https://github.com/rapidsai/cudf/tree/main/python/dask_cudf) for more information.

## When to use cuDF and Dask cuDF[#](#when-to-use-cudf-and-dask-cudf "Permalink to this heading")

If your workflow is fast enough on a single GPU or your data comfortably fits in memory on a single GPU, you would want to use cuDF. If you want to distribute your workflow across multiple GPUs, have more data than you can fit in memory on a single GPU, or want to analyze data spread across many files at once, you would want to use Dask cuDF.

import os

import cupy as cp
import pandas as pd

import cudf
import dask\_cudf

cp.random.seed(12)

\#### Portions of this were borrowed and adapted from the
\#### cuDF cheatsheet, existing cuDF documentation,
\#### and 10 Minutes to Pandas.

## Object Creation[#](#object-creation "Permalink to this heading")

Creating a `cudf.Series` and `dask_cudf.Series`.

s \= cudf.Series(\[1, 2, 3, None, 4\])
s

0       1
1       2
2       3
3    <NA>
4       4
dtype: int64

ds \= dask\_cudf.from\_cudf(s, npartitions\=2)
\# Note the call to head here to show the first few entries, unlike
\# cuDF objects, Dask-cuDF objects do not have a printing
\# representation that shows values since they may not be in local
\# memory.
ds.head(n\=3)

0    1
1    2
2    3
dtype: int64

Creating a `cudf.DataFrame` and a `dask_cudf.DataFrame` by specifying values for each column.

df \= cudf.DataFrame(
    {
        "a": list(range(20)),
        "b": list(reversed(range(20))),
        "c": list(range(20)),
    }
)
df

a

b

c

0

0

19

0

1

1

18

1

2

2

17

2

3

3

16

3

4

4

15

4

5

5

14

5

6

6

13

6

7

7

12

7

8

8

11

8

9

9

10

9

10

10

9

10

11

11

8

11

12

12

7

12

13

13

6

13

14

14

5

14

15

15

4

15

16

16

3

16

17

17

2

17

18

18

1

18

19

19

0

19

Now we will convert our cuDF dataframe into a Dask-cuDF equivalent. Here we call out a key difference: to inspect the data we must call a method (here `.head()` to look at the first few values). In the general case (see the end of this notebook), the data in `ddf` will be distributed across multiple GPUs.

In this small case, we could call `ddf.compute()` to obtain a cuDF object from the Dask-cuDF object. In general, we should avoid calling `.compute()` on large dataframes, and restrict ourselves to using it when we have some (relatively) small postprocessed result that we wish to inspect. Hence, throughout this notebook we will generally call `.head()` to inspect the first few values of a Dask-cuDF dataframe, occasionally calling out places where we use `.compute()` and why.

_To understand more of the differences between how cuDF and Dask cuDF behave here, visit the [10 Minutes to Dask](https://docs.dask.org/en/stable/10-minutes-to-dask.html) tutorial after this one._

ddf \= dask\_cudf.from\_cudf(df, npartitions\=2)
ddf.head()

a

b

c

0

0

19

0

1

1

18

1

2

2

17

2

3

3

16

3

4

4

15

4

Creating a `cudf.DataFrame` from a pandas `Dataframe` and a `dask_cudf.Dataframe` from a `cudf.Dataframe`.

_Note that best practice for using dask-cuDF is to read data directly into a `dask_cudf.DataFrame` with `read_csv` or other builtin I/O routines (discussed below)._

pdf \= pd.DataFrame({"a": \[0, 1, 2, 3\], "b": \[0.1, 0.2, None, 0.3\]})
gdf \= cudf.DataFrame.from\_pandas(pdf)
gdf

a

b

0

0

0.1

1

1

0.2

2

2

<NA>

3

3

0.3

dask\_gdf \= dask\_cudf.from\_cudf(gdf, npartitions\=2)
dask\_gdf.head(n\=2)

a

b

0

0

0.1

1

1

0.2

## Viewing Data[#](#viewing-data "Permalink to this heading")

Viewing the top rows of a GPU dataframe.

df.head(2)

a

b

c

0

0

19

0

1

1

18

1

ddf.head(2)

a

b

c

0

0

19

0

1

1

18

1

Sorting by values.

df.sort\_values(by\="b")

a

b

c

19

19

0

19

18

18

1

18

17

17

2

17

16

16

3

16

15

15

4

15

14

14

5

14

13

13

6

13

12

12

7

12

11

11

8

11

10

10

9

10

9

9

10

9

8

8

11

8

7

7

12

7

6

6

13

6

5

5

14

5

4

4

15

4

3

3

16

3

2

2

17

2

1

1

18

1

0

0

19

0

ddf.sort\_values(by\="b").head()

a

b

c

19

19

0

19

18

18

1

18

17

17

2

17

16

16

3

16

15

15

4

15

## Selecting a Column[#](#selecting-a-column "Permalink to this heading")

Selecting a single column, which initially yields a `cudf.Series` or `dask_cudf.Series`. Calling `compute` results in a `cudf.Series` (equivalent to `df.a`).

df\["a"\]

0      0
1      1
2      2
3      3
4      4
5      5
6      6
7      7
8      8
9      9
10    10
11    11
12    12
13    13
14    14
15    15
16    16
17    17
18    18
19    19
Name: a, dtype: int64

ddf\["a"\].head()

0    0
1    1
2    2
3    3
4    4
Name: a, dtype: int64

## Selecting Rows by Label[#](#selecting-rows-by-label "Permalink to this heading")

Selecting rows from index 2 to index 5 from columns ‘a’ and ‘b’.

df.loc\[2:5, \["a", "b"\]\]

a

b

2

2

17

3

3

16

4

4

15

5

5

14

ddf.loc\[2:5, \["a", "b"\]\].head()

a

b

2

2

17

3

3

16

4

4

15

5

5

14

## Selecting Rows by Position[#](#selecting-rows-by-position "Permalink to this heading")

Selecting via integers and integer slices, like numpy/pandas. Note that this functionality is not available for Dask-cuDF DataFrames.

df.iloc\[0\]

a     0
b    19
c     0
Name: 0, dtype: int64

df.iloc\[0:3, 0:2\]

a

b

0

0

19

1

1

18

2

2

17

You can also select elements of a `DataFrame` or `Series` with direct index access.

df\[3:5\]

a

b

c

3

3

16

3

4

4

15

4

s\[3:5\]

3    <NA>
4       4
dtype: int64

## Boolean Indexing[#](#boolean-indexing "Permalink to this heading")

Selecting rows in a `DataFrame` or `Series` by direct Boolean indexing.

df\[df.b \> 15\]

a

b

c

0

0

19

0

1

1

18

1

2

2

17

2

3

3

16

3

ddf\[ddf.b \> 15\].head(n\=3)

a

b

c

0

0

19

0

1

1

18

1

2

2

17

2

Selecting values from a `DataFrame` where a Boolean condition is met, via the `query` API.

df.query("b == 3")

a

b

c

16

16

3

16

Note here we call `compute()` rather than `head()` on the Dask-cuDF dataframe since we are happy that the number of matching rows will be small (and hence it is reasonable to bring the entire result back).

ddf.query("b == 3").compute()

a

b

c

16

16

3

16

You can also pass local variables to Dask-cuDF queries, via the `local_dict` keyword. With standard cuDF, you may either use the `local_dict` keyword or directly pass the variable via the `@` keyword. Supported logical operators include `>`, `<`, `>=`, `<=`, `==`, and `!=`.

cudf\_comparator \= 3
df.query("b == @cudf\_comparator")

a

b

c

16

16

3

16

dask\_cudf\_comparator \= 3
ddf.query("b == @val", local\_dict\={"val": dask\_cudf\_comparator}).compute()

a

b

c

16

16

3

16

Using the `isin` method for filtering.

df\[df.a.isin(\[0, 5\])\]

a

b

c

0

0

19

0

5

5

14

5

## MultiIndex[#](#multiindex "Permalink to this heading")

cuDF supports hierarchical indexing of DataFrames using MultiIndex. Grouping hierarchically (see `Grouping` below) automatically produces a DataFrame with a MultiIndex.

arrays \= \[\["a", "a", "b", "b"\], \[1, 2, 3, 4\]\]
tuples \= list(zip(\*arrays))
idx \= cudf.MultiIndex.from\_tuples(tuples)
idx

MultiIndex(\[('a', 1),
            ('a', 2),
            ('b', 3),
            ('b', 4)\],
           )

This index can back either axis of a DataFrame.

gdf1 \= cudf.DataFrame(
    {"first": cp.random.rand(4), "second": cp.random.rand(4)}
)
gdf1.index \= idx
gdf1

first

second

a

1

0.082654

0.967955

2

0.399417

0.441425

b

3

0.784297

0.793582

4

0.070303

0.271711

gdf2 \= cudf.DataFrame(
    {"first": cp.random.rand(4), "second": cp.random.rand(4)}
).T
gdf2.columns \= idx
gdf2

a

b

1

2

3

4

first

0.343382

0.003700

0.20043

0.581614

second

0.907812

0.101512

0.24179

0.224180

Accessing values of a DataFrame with a MultiIndex, both with `.loc`

gdf1.loc\[("b", 3)\]

first     0.784297
second    0.793582
Name: ('b', 3), dtype: float64

And `.iloc`

gdf1.iloc\[0:2\]

first

second

a

1

0.082654

0.967955

2

0.399417

0.441425

## Missing Data[#](#missing-data "Permalink to this heading")

Missing data can be replaced by using the `fillna` method.

s.fillna(999)

0      1
1      2
2      3
3    999
4      4
dtype: int64

ds.fillna(999).head(n\=3)

0    1
1    2
2    3
dtype: int64

## Stats[#](#stats "Permalink to this heading")

Calculating descriptive statistics for a `Series`.

s.mean(), s.var()

(np.float64(2.5), np.float64(1.666666666666666))

This serves as a prototypical example of when we might want to call `.compute()`. The result of computing the mean and variance is a single number in each case, so it is definitely reasonable to look at the entire result!

ds.mean().compute(), ds.var().compute()

(np.float64(2.5), np.float64(1.6666666666666667))

## Applymap[#](#applymap "Permalink to this heading")

Applying functions to a `Series`. Note that applying user defined functions directly with Dask cuDF is not yet implemented. For now, you can use [map\_partitions](http://docs.dask.org/en/stable/generated/dask.dataframe.DataFrame.map_partitions.html) to apply a function to each partition of the distributed dataframe.

def add\_ten(num):
    return num + 10

df\["a"\].apply(add\_ten)

0     10
1     11
2     12
3     13
4     14
5     15
6     16
7     17
8     18
9     19
10    20
11    21
12    22
13    23
14    24
15    25
16    26
17    27
18    28
19    29
Name: a, dtype: int64

ddf\["a"\].map\_partitions(add\_ten).head()

0    10
1    11
2    12
3    13
4    14
Name: a, dtype: int64

## Histogramming[#](#histogramming "Permalink to this heading")

Counting the number of occurrences of each unique value of variable.

df.a.value\_counts()

a
15    1
10    1
18    1
2     1
11    1
5     1
3     1
16    1
9     1
12    1
19    1
6     1
7     1
8     1
13    1
4     1
17    1
14    1
1     1
0     1
Name: count, dtype: int64

ddf.a.value\_counts().head()

a
6     1
17    1
12    1
16    1
7     1
Name: count, dtype: int64

## String Methods[#](#string-methods "Permalink to this heading")

Like pandas, cuDF provides string processing methods in the `str` attribute of `Series`. Full documentation of string methods is a work in progress. Please see the [cuDF API documentation](https://docs.rapids.ai/api/cudf/stable/api_docs/series.html#string-handling) for more information.

s \= cudf.Series(\["A", "B", "C", "Aaba", "Baca", None, "CABA", "dog", "cat"\])
s.str.lower()

0       a
1       b
2       c
3    aaba
4    baca
5    <NA>
6    caba
7     dog
8     cat
dtype: object

ds \= dask\_cudf.from\_cudf(s, npartitions\=2)
ds.str.lower().head(n\=4)

0       a
1       b
2       c
3    aaba
dtype: object

As well as simple manipulation, We can also match strings using [regular expressions](https://docs.rapids.ai/api/cudf/stable/api_docs/api/cudf.core.column.string.StringMethods.match.html).

s.str.match("^\[aAc\].+")

0    False
1    False
2    False
3     True
4    False
5     <NA>
6    False
7    False
8     True
dtype: bool

ds.str.match("^\[aAc\].+").head()

0    False
1    False
2    False
3     True
4    False
dtype: bool

## Concat[#](#concat "Permalink to this heading")

Concatenating `Series` and `DataFrames` row-wise.

s \= cudf.Series(\[1, 2, 3, None, 5\])
cudf.concat(\[s, s\])

0       1
1       2
2       3
3    <NA>
4       5
0       1
1       2
2       3
3    <NA>
4       5
dtype: int64

ds2 \= dask\_cudf.from\_cudf(s, npartitions\=2)
dask\_cudf.concat(\[ds2, ds2\]).head(n\=3)

0    1
1    2
2    3
dtype: int64

## Join[#](#join "Permalink to this heading")

Performing SQL style merges. Note that the dataframe order is **not maintained**, but may be restored post-merge by sorting by the index.

df\_a \= cudf.DataFrame()
df\_a\["key"\] \= \["a", "b", "c", "d", "e"\]
df\_a\["vals\_a"\] \= \[float(i + 10) for i in range(5)\]

df\_b \= cudf.DataFrame()
df\_b\["key"\] \= \["a", "c", "e"\]
df\_b\["vals\_b"\] \= \[float(i + 100) for i in range(3)\]

merged \= df\_a.merge(df\_b, on\=\["key"\], how\="left")
merged

key

vals\_a

vals\_b

0

a

10.0

100.0

1

c

12.0

101.0

2

e

14.0

102.0

3

b

11.0

<NA>

4

d

13.0

<NA>

ddf\_a \= dask\_cudf.from\_cudf(df\_a, npartitions\=2)
ddf\_b \= dask\_cudf.from\_cudf(df\_b, npartitions\=2)

merged \= ddf\_a.merge(ddf\_b, on\=\["key"\], how\="left").head(n\=4)
merged

key

vals\_a

vals\_b

0

c

12.0

101.0

1

e

14.0

102.0

2

b

11.0

<NA>

3

d

13.0

<NA>

## Grouping[#](#grouping "Permalink to this heading")

Like [pandas](https://pandas.pydata.org/docs/user_guide/groupby.html), cuDF and Dask-cuDF support the [Split-Apply-Combine groupby paradigm](https://doi.org/10.18637/jss.v040.i01).

df\["agg\_col1"\] \= \[1 if x % 2 \== 0 else 0 for x in range(len(df))\]
df\["agg\_col2"\] \= \[1 if x % 3 \== 0 else 0 for x in range(len(df))\]

ddf \= dask\_cudf.from\_cudf(df, npartitions\=2)

Grouping and then applying the `sum` function to the grouped data.

df.groupby("agg\_col1").sum()

a

b

c

agg\_col2

agg\_col1

1

90

100

90

4

0

100

90

100

3

ddf.groupby("agg\_col1").sum().compute()

a

b

c

agg\_col2

agg\_col1

1

90

100

90

4

0

100

90

100

3

Grouping hierarchically then applying the `sum` function to grouped data.

df.groupby(\["agg\_col1", "agg\_col2"\]).sum()

a

b

c

agg\_col1

agg\_col2

1

1

36

40

36

0

54

60

54

0

1

27

30

27

0

73

60

73

ddf.groupby(\["agg\_col1", "agg\_col2"\]).sum().compute()

a

b

c

agg\_col1

agg\_col2

1

1

36

40

36

0

0

73

60

73

1

27

30

27

1

0

54

60

54

Grouping and applying statistical functions to specific columns, using `agg`.

df.groupby("agg\_col1").agg({"a": "max", "b": "mean", "c": "sum"})

a

b

c

agg\_col1

1

18

10.0

90

0

19

9.0

100

ddf.groupby("agg\_col1").agg({"a": "max", "b": "mean", "c": "sum"}).compute()

a

b

c

agg\_col1

1

18

10.0

90

0

19

9.0

100

## Transpose[#](#transpose "Permalink to this heading")

Transposing a dataframe, using either the `transpose` method or `T` property. Currently, all columns must have the same type. Transposing is not currently implemented in Dask cuDF.

sample \= cudf.DataFrame({"a": \[1, 2, 3\], "b": \[4, 5, 6\]})
sample

a

b

0

1

4

1

2

5

2

3

6

sample.transpose()

0

1

2

a

1

2

3

b

4

5

6

## Time Series[#](#time-series "Permalink to this heading")

`DataFrames` supports `datetime` typed columns, which allow users to interact with and filter data based on specific timestamps.

import datetime as dt

date\_df \= cudf.DataFrame()
date\_df\["date"\] \= pd.date\_range("11/20/2018", periods\=72, freq\="D")
date\_df\["value"\] \= cp.random.sample(len(date\_df))

search\_date \= dt.datetime.strptime("2018-11-23", "%Y-%m-%d")
date\_df.query("date <= @search\_date")

date

value

0

2018-11-20

0.986051

1

2018-11-21

0.232034

2

2018-11-22

0.397617

3

2018-11-23

0.103839

date\_ddf \= dask\_cudf.from\_cudf(date\_df, npartitions\=2)
date\_ddf.query(
    "date <= @search\_date", local\_dict\={"search\_date": search\_date}
).compute()

date

value

0

2018-11-20

0.986051

1

2018-11-21

0.232034

2

2018-11-22

0.397617

3

2018-11-23

0.103839

## Categoricals[#](#categoricals "Permalink to this heading")

`DataFrames` support categorical columns.

gdf \= cudf.DataFrame(
    {"id": \[1, 2, 3, 4, 5, 6\], "grade": \["a", "b", "b", "a", "a", "e"\]}
)
gdf\["grade"\] \= gdf\["grade"\].astype("category")
gdf

id

grade

0

1

a

1

2

b

2

3

b

3

4

a

4

5

a

5

6

e

dgdf \= dask\_cudf.from\_cudf(gdf, npartitions\=2)
dgdf.head(n\=3)

id

grade

0

1

a

1

2

b

2

3

b

Accessing the categories of a column. Note that this is currently not supported in Dask-cuDF.

gdf.grade.cat.categories

Index(\['a', 'b', 'e'\], dtype='object')

Accessing the underlying code values of each categorical observation.

gdf.grade.cat.codes

0    0
1    1
2    1
3    0
4    0
5    2
dtype: uint8

dgdf.grade.cat.codes.compute()

0    0
1    1
2    1
3    0
4    0
5    2
dtype: uint8

## Converting to Pandas[#](#converting-to-pandas "Permalink to this heading")

Converting a cuDF and Dask-cuDF `DataFrame` to a pandas `DataFrame`.

df.head().to\_pandas()

a

b

c

agg\_col1

agg\_col2

0

0

19

0

1

1

1

1

18

1

0

0

2

2

17

2

1

0

3

3

16

3

0

1

4

4

15

4

1

0

To convert the first few entries to pandas, we similarly call `.head()` on the Dask-cuDF dataframe to obtain a local cuDF dataframe, which we can then convert.

ddf.head().to\_pandas()

a

b

c

agg\_col1

agg\_col2

0

0

19

0

1

1

1

1

18

1

0

0

2

2

17

2

1

0

3

3

16

3

0

1

4

4

15

4

1

0

In contrast, if we want to convert the entire frame, we need to call `.compute()` on `ddf` to get a local cuDF dataframe, and then call `to_pandas()`, followed by subsequent processing. This workflow is less recommended, since it both puts high memory pressure on a single GPU (the `.compute()` call) and does not take advantage of GPU acceleration for processing (the computation happens on in pandas).

ddf.compute().to\_pandas().head()

a

b

c

agg\_col1

agg\_col2

0

0

19

0

1

1

1

1

18

1

0

0

2

2

17

2

1

0

3

3

16

3

0

1

4

4

15

4

1

0

## Converting to Numpy[#](#converting-to-numpy "Permalink to this heading")

Converting a cuDF or Dask-cuDF `DataFrame` to a numpy `ndarray`.

df.to\_numpy()

array(\[\[ 0, 19,  0,  1,  1\],
       \[ 1, 18,  1,  0,  0\],
       \[ 2, 17,  2,  1,  0\],
       \[ 3, 16,  3,  0,  1\],
       \[ 4, 15,  4,  1,  0\],
       \[ 5, 14,  5,  0,  0\],
       \[ 6, 13,  6,  1,  1\],
       \[ 7, 12,  7,  0,  0\],
       \[ 8, 11,  8,  1,  0\],
       \[ 9, 10,  9,  0,  1\],
       \[10,  9, 10,  1,  0\],
       \[11,  8, 11,  0,  0\],
       \[12,  7, 12,  1,  1\],
       \[13,  6, 13,  0,  0\],
       \[14,  5, 14,  1,  0\],
       \[15,  4, 15,  0,  1\],
       \[16,  3, 16,  1,  0\],
       \[17,  2, 17,  0,  0\],
       \[18,  1, 18,  1,  1\],
       \[19,  0, 19,  0,  0\]\])

ddf.compute().to\_numpy()

array(\[\[ 0, 19,  0,  1,  1\],
       \[ 1, 18,  1,  0,  0\],
       \[ 2, 17,  2,  1,  0\],
       \[ 3, 16,  3,  0,  1\],
       \[ 4, 15,  4,  1,  0\],
       \[ 5, 14,  5,  0,  0\],
       \[ 6, 13,  6,  1,  1\],
       \[ 7, 12,  7,  0,  0\],
       \[ 8, 11,  8,  1,  0\],
       \[ 9, 10,  9,  0,  1\],
       \[10,  9, 10,  1,  0\],
       \[11,  8, 11,  0,  0\],
       \[12,  7, 12,  1,  1\],
       \[13,  6, 13,  0,  0\],
       \[14,  5, 14,  1,  0\],
       \[15,  4, 15,  0,  1\],
       \[16,  3, 16,  1,  0\],
       \[17,  2, 17,  0,  0\],
       \[18,  1, 18,  1,  1\],
       \[19,  0, 19,  0,  0\]\])

Converting a cuDF or Dask-cuDF `Series` to a numpy `ndarray`.

df\["a"\].to\_numpy()

array(\[ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
       17, 18, 19\])

ddf\["a"\].compute().to\_numpy()

array(\[ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
       17, 18, 19\])

## Converting to Arrow[#](#converting-to-arrow "Permalink to this heading")

Converting a cuDF or Dask-cuDF `DataFrame` to a PyArrow `Table`.

df.to\_arrow()

pyarrow.Table
a: int64
b: int64
c: int64
agg\_col1: int64
agg\_col2: int64
----
a: \[\[0,1,2,3,4,...,15,16,17,18,19\]\]
b: \[\[19,18,17,16,15,...,4,3,2,1,0\]\]
c: \[\[0,1,2,3,4,...,15,16,17,18,19\]\]
agg\_col1: \[\[1,0,1,0,1,...,0,1,0,1,0\]\]
agg\_col2: \[\[1,0,0,1,0,...,1,0,0,1,0\]\]

ddf.head().to\_arrow()

pyarrow.Table
a: int64
b: int64
c: int64
agg\_col1: int64
agg\_col2: int64
----
a: \[\[0,1,2,3,4\]\]
b: \[\[19,18,17,16,15\]\]
c: \[\[0,1,2,3,4\]\]
agg\_col1: \[\[1,0,1,0,1\]\]
agg\_col2: \[\[1,0,0,1,0\]\]

## Reading/Writing CSV Files[#](#reading-writing-csv-files "Permalink to this heading")

Writing to a CSV file.

if not os.path.exists("example\_output"):
    os.mkdir("example\_output")

df.to\_csv("example\_output/foo.csv", index\=False)

ddf.compute().to\_csv("example\_output/foo\_dask.csv", index\=False)

Reading from a csv file.

df \= cudf.read\_csv("example\_output/foo.csv")
df

a

b

c

agg\_col1

agg\_col2

0

0

19

0

1

1

1

1

18

1

0

0

2

2

17

2

1

0

3

3

16

3

0

1

4

4

15

4

1

0

5

5

14

5

0

0

6

6

13

6

1

1

7

7

12

7

0

0

8

8

11

8

1

0

9

9

10

9

0

1

10

10

9

10

1

0

11

11

8

11

0

0

12

12

7

12

1

1

13

13

6

13

0

0

14

14

5

14

1

0

15

15

4

15

0

1

16

16

3

16

1

0

17

17

2

17

0

0

18

18

1

18

1

1

19

19

0

19

0

0

Note that for the Dask-cuDF case, we use `dask_cudf.read_csv` in preference to `dask_cudf.from_cudf(cudf.read_csv)` since the former can parallelize across multiple GPUs and handle larger CSV files that would fit in memory on a single GPU.

ddf \= dask\_cudf.read\_csv("example\_output/foo\_dask.csv")
ddf.head()

a

b

c

agg\_col1

agg\_col2

0

0

19

0

1

1

1

1

18

1

0

0

2

2

17

2

1

0

3

3

16

3

0

1

4

4

15

4

1

0

Reading all CSV files in a directory into a single `dask_cudf.DataFrame`, using the star wildcard.

ddf \= dask\_cudf.read\_csv("example\_output/\*.csv")
ddf.head()

a

b

c

agg\_col1

agg\_col2

0

0

19

0

1

1

1

1

18

1

0

0

2

2

17

2

1

0

3

3

16

3

0

1

4

4

15

4

1

0

## Reading/Writing Parquet Files[#](#reading-writing-parquet-files "Permalink to this heading")

Writing to parquet files with cuDF’s GPU-accelerated parquet writer

df.to\_parquet("example\_output/temp\_parquet")

Reading parquet files with cuDF’s GPU-accelerated parquet reader.

df \= cudf.read\_parquet("example\_output/temp\_parquet")
df

a

b

c

agg\_col1

agg\_col2

0

0

19

0

1

1

1

1

18

1

0

0

2

2

17

2

1

0

3

3

16

3

0

1

4

4

15

4

1

0

5

5

14

5

0

0

6

6

13

6

1

1

7

7

12

7

0

0

8

8

11

8

1

0

9

9

10

9

0

1

10

10

9

10

1

0

11

11

8

11

0

0

12

12

7

12

1

1

13

13

6

13

0

0

14

14

5

14

1

0

15

15

4

15

0

1

16

16

3

16

1

0

17

17

2

17

0

0

18

18

1

18

1

1

19

19

0

19

0

0

Writing to parquet files from a `dask_cudf.DataFrame` using cuDF’s parquet writer under the hood.

ddf.to\_parquet("example\_output/ddf\_parquet\_files")

## Reading/Writing ORC Files[#](#reading-writing-orc-files "Permalink to this heading")

Writing ORC files.

df.to\_orc("example\_output/temp\_orc")

And reading

df2 \= cudf.read\_orc("example\_output/temp\_orc")
df2

a

b

c

agg\_col1

agg\_col2

0

0

19

0

1

1

1

1

18

1

0

0

2

2

17

2

1

0

3

3

16

3

0

1

4

4

15

4

1

0

5

5

14

5

0

0

6

6

13

6

1

1

7

7

12

7

0

0

8

8

11

8

1

0

9

9

10

9

0

1

10

10

9

10

1

0

11

11

8

11

0

0

12

12

7

12

1

1

13

13

6

13

0

0

14

14

5

14

1

0

15

15

4

15

0

1

16

16

3

16

1

0

17

17

2

17

0

0

18

18

1

18

1

1

19

19

0

19

0

0

## Dask Performance Tips[#](#dask-performance-tips "Permalink to this heading")

Like Apache Spark, Dask operations are [lazy](https://en.wikipedia.org/wiki/Lazy_evaluation). Instead of being executed immediately, most operations are added to a task graph and the actual evaluation is delayed until the result is needed.

Sometimes, though, we want to force the execution of operations. Calling `persist` on a Dask collection fully computes it (or actively computes it in the background), persisting the result into memory. When we’re using distributed systems, we may want to wait until `persist` is finished before beginning any downstream operations. We can enforce this contract by using `wait`. Wrapping an operation with `wait` will ensure it doesn’t begin executing until all necessary upstream operations have finished.

The snippets below provide basic examples, using `LocalCUDACluster` to create one dask-worker per GPU on the local machine. For more detailed information about `persist` and `wait`, please see the Dask documentation for [persist](https://docs.dask.org/en/latest/api.html#dask.persist) and [wait](https://docs.dask.org/en/latest/futures.html#distributed.wait). Wait relies on the concept of Futures, which is beyond the scope of this tutorial. For more information on Futures, see the Dask [Futures](https://docs.dask.org/en/latest/futures.html) documentation. For more information about multi-GPU clusters, please see the [dask-cuda](https://github.com/rapidsai/dask-cuda) library (documentation is in progress).

First, we set up a GPU cluster. With our `client` set up, Dask-cuDF computation will be distributed across the GPUs in the cluster.

import time

from dask.distributed import Client, wait
from dask\_cuda import LocalCUDACluster

cluster \= LocalCUDACluster()
client \= Client(cluster)

### Persisting Data[#](#persisting-data "Permalink to this heading")

Next, we create our Dask-cuDF DataFrame and apply a transformation, storing the result as a new column.

nrows \= 10000000

df2 \= cudf.DataFrame({"a": cp.arange(nrows), "b": cp.arange(nrows)})
ddf2 \= dask\_cudf.from\_cudf(df2, npartitions\=16)
ddf2\["c"\] \= ddf2\["a"\] + 5
ddf2

**Dask DataFrame Structure:**

a

b

c

npartitions=16

0

int64

int64

int64

625000

...

...

...

...

...

...

...

9375000

...

...

...

9999999

...

...

...

Dask Name: assign, 4 expressions

!nvidia-smi

Thu Oct 10 06:44:46 2024       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 550.54.15              Driver Version: 550.54.15      CUDA Version: 12.5     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  Tesla V100-PCIE-32GB           Off |   00000000:06:00.0 Off |                    0 |
| N/A   32C    P0             36W /  250W |     643MiB /  32768MiB |      1%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
                                                                                         
+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI        PID   Type   Process name                              GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
+-----------------------------------------------------------------------------------------+

Because Dask is lazy, the computation has not yet occurred. We can see that there are sixty-four tasks in the task graph and we’re using about 330 MB of device memory on each GPU. We can force computation by using `persist`. By forcing execution, the result is now explicitly in memory and our task graph only contains one task per partition (the baseline).

ddf2 \= ddf2.persist()
ddf2

/opt/conda/envs/docs/lib/python3.11/site-packages/distributed/client.py:3361: UserWarning: Sending large graph of size 152.61 MiB.
This may cause some slowdown.
Consider loading the data with Dask directly
 or using futures or delayed objects to embed the data into the graph without repetition.
See also https://docs.dask.org/en/stable/best-practices.html#load-data-with-dask for more information.
  warnings.warn(

**Dask DataFrame Structure:**

a

b

c

npartitions=16

0

int64

int64

int64

625000

...

...

...

...

...

...

...

9375000

...

...

...

9999999

...

...

...

Dask Name: getitem-add-assign, 1 expression

\# Sleep to ensure the persist finishes and shows in the memory usage
!sleep 5; nvidia-smi

Thu Oct 10 06:44:52 2024       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 550.54.15              Driver Version: 550.54.15      CUDA Version: 12.5     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  Tesla V100-PCIE-32GB           Off |   00000000:06:00.0 Off |                    0 |
| N/A   32C    P0             36W /  250W |    1433MiB /  32768MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
                                                                                         
+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI        PID   Type   Process name                              GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
+-----------------------------------------------------------------------------------------+

Because we forced computation, we now have a larger object in distributed GPU memory. Note that actual numbers will differ between systems (for example depending on how many devices are available).

### Wait[#](#wait "Permalink to this heading")

Depending on our workflow or distributed computing setup, we may want to `wait` until all upstream tasks have finished before proceeding with a specific function. This section shows an example of this behavior, adapted from the Dask documentation.

First, we create a new Dask DataFrame and define a function that we’ll map to every partition in the dataframe.

import random

nrows \= 10000000

df1 \= cudf.DataFrame({"a": cp.arange(nrows), "b": cp.arange(nrows)})
ddf1 \= dask\_cudf.from\_cudf(df1, npartitions\=100)

def func(df):
    time.sleep(random.randint(1, 10))
    return (df + 5) \* 3 \- 11

This function will do a basic transformation of every column in the dataframe, but the time spent in the function will vary due to the `time.sleep` statement randomly adding 1-10 seconds of time. We’ll run this on every partition of our dataframe using `map_partitions`, which adds the task to our task-graph, and store the result. We can then call `persist` to force execution.

results\_ddf \= ddf2.map\_partitions(func)
results\_ddf \= results\_ddf.persist()

However, some partitions will be done **much** sooner than others. If we had downstream processes that should wait for all partitions to be completed, we can enforce that behavior using `wait`.

wait(results\_ddf)

DoneAndNotDoneFutures(done={<Future: finished, type: cudf.core.dataframe.DataFrame, key: ('func-6ce8202a759c47ce931df134df0dbe7e', 4)>, <Future: finished, type: cudf.core.dataframe.DataFrame, key: ('func-6ce8202a759c47ce931df134df0dbe7e', 12)>, <Future: finished, type: cudf.core.dataframe.DataFrame, key: ('func-6ce8202a759c47ce931df134df0dbe7e', 2)>, <Future: finished, type: cudf.core.dataframe.DataFrame, key: ('func-6ce8202a759c47ce931df134df0dbe7e', 3)>, <Future: finished, type: cudf.core.dataframe.DataFrame, key: ('func-6ce8202a759c47ce931df134df0dbe7e', 8)>, <Future: finished, type: cudf.core.dataframe.DataFrame, key: ('func-6ce8202a759c47ce931df134df0dbe7e', 13)>, <Future: finished, type: cudf.core.dataframe.DataFrame, key: ('func-6ce8202a759c47ce931df134df0dbe7e', 10)>, <Future: finished, type: cudf.core.dataframe.DataFrame, key: ('func-6ce8202a759c47ce931df134df0dbe7e', 11)>, <Future: finished, type: cudf.core.dataframe.DataFrame, key: ('func-6ce8202a759c47ce931df134df0dbe7e', 14)>, <Future: finished, type: cudf.core.dataframe.DataFrame, key: ('func-6ce8202a759c47ce931df134df0dbe7e', 15)>, <Future: finished, type: cudf.core.dataframe.DataFrame, key: ('func-6ce8202a759c47ce931df134df0dbe7e', 6)>, <Future: finished, type: cudf.core.dataframe.DataFrame, key: ('func-6ce8202a759c47ce931df134df0dbe7e', 7)>, <Future: finished, type: cudf.core.dataframe.DataFrame, key: ('func-6ce8202a759c47ce931df134df0dbe7e', 1)>, <Future: finished, type: cudf.core.dataframe.DataFrame, key: ('func-6ce8202a759c47ce931df134df0dbe7e', 0)>, <Future: finished, type: cudf.core.dataframe.DataFrame, key: ('func-6ce8202a759c47ce931df134df0dbe7e', 9)>, <Future: finished, type: cudf.core.dataframe.DataFrame, key: ('func-6ce8202a759c47ce931df134df0dbe7e', 5)>}, not\_done=set())

With `wait` completed, we can safely proceed on in our workflow.

On this page

[Show Source](../../_sources/user_guide/10min.ipynb.txt)