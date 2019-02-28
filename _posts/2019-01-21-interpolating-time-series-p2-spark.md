---
title: >-
    Interpolating Time Series Data in Apache Spark and Python Pandas - Part 2: PySpark
tags:
  - time series data
  - python
  - pyspark
excerpt: Introducing end-to-end time series interpolation in PySpark.
---

This is the second out of two posts about interpolating time series data using PySpark and Python Pandas. In [this post](https://walkenho.github.io/interpolating-time-series-p1-pandas/) last week, we
covered how to use three different interpolation methods (forward filling, backward filling, interpolation) in Pandas,
this week, we will cover how we can implement the same functionality in PySpark. 
The full code for this post can be found [here in my github](https://github.com/walkenho/tales-of-1001-data/blob/master/timeseries-interpolation-in-spark/interpolating_timeseries_p2_pyspark.ipynb).

## Preparing the Data and Visualization of the Problem

We follow the same procedure as last week to generate a data set with missing values to interpolate. First we generate a pandas data frame with some test data. The data set contains data for two houses and uses a $$sin()$$ and a $$cos()$$ function to generate some read data for a set of dates. To generate the missing values, we randomly drop half of the entries.


```python
import pandas as pd
import numpy as np
import random
data = {'readtime': pd.date_range(start='1/15/2018', end='02/14/2018', freq='D')\
                .append(pd.date_range(start='1/15/2018', end='02/14/2018', freq='D')),
        'house' : ['house1' for i in range(31)] + ['house2' for i in range(31)],
        'readvalue': [0.5+0.5*np.sin(2*np.pi/30*i) for i in range(31)]\
                + [0.5+0.5*np.cos(2*np.pi/30*i) for i in range(31)]}
df0 = pd.DataFrame(data, columns = ['readtime', 'house', 'readvalue'])

random.seed(42)
df0 = df0.drop(random.sample(range(df0.shape[0]), k=int(df0.shape[0]/2)))
df0.head()
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>readtime</th>
      <th>house</th>
      <th>readvalue</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2018-01-15</td>
      <td>house1</td>
      <td>0.500000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2018-01-18</td>
      <td>house1</td>
      <td>0.793893</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2018-01-19</td>
      <td>house1</td>
      <td>0.871572</td>
    </tr>
    <tr>
      <th>9</th>
      <td>2018-01-24</td>
      <td>house1</td>
      <td>0.975528</td>
    </tr>
    <tr>
      <th>10</th>
      <td>2018-01-25</td>
      <td>house1</td>
      <td>0.933013</td>
    </tr>
  </tbody>
</table>
</div>



The following graph shows the data with the missing values clearly visible.

<figure class="align-center">
  <img src="{{ site.url }}{{ site.baseurl }}/images/interpolating-time-series-p1-pandas-fig1.png" alt="">
    <figcaption>Read Data with Missing Entries</figcaption>
</figure>


In order to work with PySpark, we convert the Pandas data frame into a Spark data frame. We need to divide the datetime by 10^9 since the unit of time is different for pandas datetime and spark. We also add the column 'readtime_existent' to keep track of which values are missing and which are not.


```python
import pyspark.sql.functions as func
from pyspark.sql.functions import col

df = spark.createDataFrame(df0)
df = df.withColumn("readtime", col('readtime')/1e9)\
        .withColumn("readtime_existent", col("readtime"))
```

We get a table like this:

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>readtime</th>
      <th>house</th>
      <th>readvalue</th>
      <th>readtime_existent</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>1.5159744E9</td>
      <td>house1</td>
      <td>0.500000</td>
      <td>1.5159744E9</td>
    </tr>
    <tr>
      <td>1.5162336E9</td>
      <td>house1</td>
      <td>0.793893</td>
      <td>1.5162336E9</td>
    </tr>
    <tr>
      <td>1.51632E9</td>
      <td>house1</td>
      <td>0.871572</td>
      <td>1.51632E9</td>
    </tr>
  </tbody>
</table>
</div>


## Interpolation
### Resampling the Read Datetime

As in Pandas, the first step is to resample the time data. However, unfortunately Spark does not provide an equivalent to 
Pandas `resample()` method. Our workaround is generating an array containing an equally spaced time grid between the mininmum 
and maximum time. The trick here is to first group the read data by house, then create the respective array for each house and 
use the sql function `explode()` to convert the array into a column. The resulting structure is then used as basis to which we add 
the read value information for the times where it exists using a left outer join. The following code shows how this is done. 


```python
from pyspark.sql.types import *

# define function to create date range
def date_range(t1, t2, step=60*60*24):
    """Returns a list of equally spaced points between t1 and t2 with stepsize step."""
    return [t1 + step*x for x in range(int((t2-t1)/step)+1)]

# define udf
date_range_udf = func.udf(date_range, ArrayType(LongType()))

# group data by house, obtain min and max time by house, create time arrays and explode them
df_base = \
    df.groupBy('house')\
        .agg(func.min('readtime').cast('integer').alias('readtime_min'), func.max('readtime').cast('integer').alias('readtime_max'))\
        .withColumn("readtime", func.explode(date_range_udf("readtime_min", "readtime_max")))\
        .drop('readtime_min', 'readtime_max')

# left outer join existing read values
df_all_dates = df_base.join(df, ["house", "readtime"], "leftouter")
```

An extract of the resulting table looks like this:

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>readtime</th>
      <th>house</th>
      <th>readvalue</th>
      <th>readtime_existent</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>1516924800</td>
      <td>house2</td>
      <td>0.165435</td>
      <td>1.5169248E9</td>
    </tr>
    <tr>
      <td>1516060800</td>
      <td>house2</td>
      <td>null</td>
      <td>null</td>
    </tr>
    <tr>
      <td>1516665600</td>
      <td>house2</td>
      <td>0.447736</td>
      <td>1.5166656E9</td>
    </tr>
  </tbody>
</table>
</div>

As one can see, a null in the readtime_existent column indicates a missing read value.


### Forward-fill and Backward-fill Using Window Functions
When using a forward-fill, we fill-in the missing data with the latest known value before now. In contrast, when using a backwards-fill, we fill-in the data with the next known value. This idea of looking at a subset of data is well known and used in SQL and can be implemented using the `pyspark.sql.Window` function in combination with `last()` and `first()`. The crucial part in both cases is to use the `ignorenulls=True` argument. On the partitioned and sorted data, we look for:

* forward-fill: the last not-null value in the window ranging from minus infinity to now
* backward-fill: the first not-null value in the window ranging from now to plus infinity

Here is how to create the interpolated columns and add them to the dataframe. 
We also keep the interpolated read times since we will need them for the interpolation to calculate the time-difference for 
which we need to interpolate. 
Looking at the code below, it becomes clear, why we needed to keep the readtime_existent column from the previous steps. If we
only had the interpolated read times, we could not figure out how big this gap was.

```python
from pyspark.sql import Window
import sys

window_ff = Window.partitionBy('house')\
               .orderBy('readtime')\
               .rowsBetween(-sys.maxsize, 0)
               
window_bf = Window.partitionBy('house')\
               .orderBy('readtime')\
               .rowsBetween(0, sys.maxsize)
        
# create the series containing the filled values
read_last = func.last(df_all_dates['readvalue'], ignorenulls=True).over(window_ff)
readtime_last = func.last(df_all_dates['readtime_existent'], ignorenulls=True).over(window_ff)

read_next = func.first(df_all_dates['readvalue'], ignorenulls=True).over(window_bf)
readtime_next = func.first(df_all_dates['readtime_existent'], ignorenulls=True).over(window_bf)

# add the columns to the dataframe
df_filled = df_all_dates.withColumn('readvalue_ff', read_last)\
                        .withColumn('readtime_ff', readtime_last)\
                        .withColumn('readvalue_bf', read_next)\
                        .withColumn('readtime_bf', readtime_next)
```

### Interpolation

Finally we use the forward filled and backwards filled data to interpolate both read datetimes and read values using a simple spline. 
This can be done using a user-defined function (if you want to learn more about how to create UDFs, you can take a look at my post [here](https://walkenho.github.io/how-to-convert-python-functions-into-pyspark-UDFs)).

```python
# define interpolation function
def interpol(x, x_prev, x_next, y_prev, y_next, y):
    if x_prev == x_next:
        return y
    else:
        m = (y_next-y_prev)/(x_next-x_prev)
        y_interpol = y_prev + m * (x - x_prev)
        return y_interpol

# convert function to udf
interpol_udf = func.udf(interpol, FloatType())   
    
# add interpolated columns to dataframe and clean up
df_filled = df_filled.withColumn('readvalue_interpol', interpol_udf('readtime', 'readtime_ff', 'readtime_bf', 'readvalue_ff', 'readvalue_bf', 'readvalue'))\
                    .drop('readtime_existent', 'readtime_ff', 'readtime_bf')\
                    .withColumnRenamed('reads_all', 'readvalue')\
                    .withColumn('readtime', func.from_unixtime(col('readtime')))
```

This leaves us with a single dataframe containing all of the interpolation methods. This is how its structure looks like:

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>house</th>
      <th>readtime</th>
      <th>readvalue</th>
      <th>readvalue_ff</th>
      <th>readvalue_bf</th>
      <th>readvalue_interpol</th>
    </tr>
  </thead>
  <tbody>
    <tr>
    <td>house1</td>
    <td>2018-01-15 00:00:00</td>
    <td>0.5</td>
    <td>0.5</td>
    <td>0.5</td>
    <td>0.5</td>
    </tr>
    <tr>
    <td>house1</td>
    <td>2018-01-16 00:00:00</td>
    <td>null</td>
    <td>0.5</td>
    <td>0.793893</td>
    <td>0.597964</td>
    </tr>
    <tr>
    <td>house1</td>
    <td>2018-01-17 00:00:00</td>
    <td>null</td>
    <td>0.5</td>
    <td>0.793893</td>
    <td>0.695928</td>
    </tr>
    <tr>
    <td>house1</td>
    <td>2018-01-17 00:00:00</td>
    <td>0.793893</td>
    <td>0.793893</td>
    <td>0.793893</td>
    <td>0.793893</td>
    </tr>
  </tbody>
</table>
</div>


Finally we can visualize the results to observe the differences between the interpolation techniques. 
The opaque dots show the interpolated values.

<figure class="align-center">
  <img src="{{ site.url }}{{ site.baseurl }}/images/interpolating-time-series-p1-pandas-fig2.png" alt="">
    <figcaption>Original data (dark) and interpolated data (light), interpolated using (top) forward filling, (middle)
    backward filling and (bottom) interpolation.</figcaption>
</figure>

We can clearly see how in the top figure, the gaps have been filled with the last known value, in the middle figure, 
the gaps have been filled with the next value to come and in the bottom figure the difference has been interpolated.

## Summary and Conclusion
In this post we have seen how we can use PySpark to perform end-to-end interpolation of time series data. We have demonstrated,
how we can use resample time series data and how we can use the 
`*Window` function in combination with the `first()` and `last()` 
function to fill-in the generated missing values. We have then seen, how we can use a user-defined function to perform a simple
spline-interpolation.

I hope this post helps to plug the gap of literature about end-to-end time series interpolation and does provide some usefulness 
for the readers.
