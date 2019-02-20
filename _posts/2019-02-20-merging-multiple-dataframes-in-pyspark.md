---
title: Merging Multiple DataFrames in PySpark
tags:
  - python
  - pyspark
excerpt: How to merge multiple dataframes in PySpark using a combination of unionAll and reduce.
---

Here is another tiny episode in the series "How to do things in PySpark", which I have apparently started.

A colleague recently asked me if I had a good way of merging multiple PySpark dataframes into a single dataframe. Since the *unionAll()* function
only accepts two arguments, a small of a workaround is needed. So, here is a short write-up of an idea that I stolen from 
[here](https://datascience.stackexchange.com/questions/11356/merging-multiple-data-frames-row-wise-in-pyspark). As always, the code has been tested for Spark 2.1.1.

The idea is to use the *unionAll()* function in combination with the *reduce()* function from the functools module. 
*reduce()* takes two arguments, a function and the input arguments for the
function. Instead of two input arguments, we can provide a list. In this case, reduce will apply the function subsequently 
to the list. An example:
```python
from functools import reduce

numbers = list(range(5))
diff = reduce(lambda x, y: x - y, numbers)
print(diff)
 
10
```
since ((((0-1)-2)-3)-4) = 10).

We can now combine this with *unionAll()* as follows.
```python
# import modules
from functools import reduce
from pyspark.sql import DataFrame

# define dataframes
df1 = sc.parallelize([[1., 'age 18-25']]).toDF(["f1", "age"])
df2 = sc.parallelize([[2., 'age 26-30']]).toDF(["f1", "age"])
df3 = sc.parallelize([[3., 'age 31-35']]).toDF(["f1", "age"])

# create list of dataframes
dfs = [df1, df2, df3]

# create merged dataframe
df_complete = reduce(DataFrame.unionAll, dfs)

df_complete.show()
```
returns
  
| f1|      age|  
|---|---------| 
|1.0|age 18-25|  
|2.0|age 26-30|
|3.0|age 31-35|

A word of caution! *unionAll* does not re-sort columns, so when you apply the procedure described above, make sure that your dataframes have the same order of columns. Otherwise you will end up with your entries in the wrong columns. 

I hope that helps :)
