---
title: >-
    Interpolating Time Series Data in Apache Spark and Python Pandas - Part 1: Pandas
tags:
  - time series data
  - python
  - pandas
overlay_image: /images/interpolating-timeseries-p1-overlay.jpg
overlay_filter: 0.5 # same as adding an opacity of 0.5 to a black background
caption: "Photo credit: [**Unsplash**](https://unsplash.com)"
actions:
  - label: "More Info"
    url: "https://unsplash.com"
excerpt: Introducing time series interpolation in Python Pandas.
---

Note: Pandas version 0.20.1 (May 2017) changed the grouping API. This post reflects the functionality of the updated version.

As we have all experienced (and I am sure suffered), real-world data is patchy most of the time and therefore a common first step in any data science task is data cleaning. Having recently moved from Pandas to Pyspark, I was used to the conveniences Pandas offers, which given to its distributed nature Pyspark is sometimes lacking. One of these features I have particularly been  missing recently is a straight-forward way of interpolating (or in-filling) time series data. Whilst the problem of in-filling missing values has been covered a few times (e.g. [here](https://johnpaton.net/posts/forward-fill-spark/)), I was not able to find a source, which detailed the end-to-end process of generating the underlying time-grid and then subsequently filling in the missing values. This and the next post try to close this gap by detailing the process in both Pandas and Pyspark. This weeks post starts by demonstrating the process using Pandas data frames. Next week's post will show how to achieve the same functionality in PySpark. The full notebook for this post can be found [here in my github](https://github.com/walkenho/tales-of-1001-data/blob/master/timeseries-interpolation-in-spark/interpolating_time_series_p1_pandas.ipynb).

## Preparing the Data and Initial Visualization

First we generate a pandas data frame df0 with some test data. We create a data set containing two houses and use a
$$sin$$ and a $$cos$$ function to generate some read data for a set of dates. To generate the missing values, we randomly drop half of the entries. 

```python
data = {'datetime': pd.date_range(start='1/15/2018', end='02/14/2018', freq='D')\
                .append(pd.date_range(start='1/15/2018', end='02/14/2018', freq='D')),
        'house' : ['house1' for i in range(31)] + ['house2' for i in range(31)],
        'readvalue':  [0.5 + 0.5*np.sin(2*np.pi/30*i) for i in range(31)]\
                    + [0.5 + 0.5*np.cos(2*np.pi/30*i) for i in range(31)]}
df0 = pd.DataFrame(data, columns = ['datetime', 'house', 'readvalue'])

# Randomly drop half the reads
random.seed(42)
df0 = df0.drop(random.sample(range(df0.shape[0]), k=int(df0.shape[0]/2)))
```

This is how the data looks like. A $$sin$$ and a $$cos$$ with plenty of missing data points.

<figure class="align-center">
  <img src="{{ site.url }}{{ site.baseurl }}/images/interpolating-timeseries-p1-pandas-fig1.png" alt="">
    <figcaption>Read Data with Missing Entries</figcaption>
</figure>

We will now look at three different methods of interpolating the missing read values: forward-filling, backward-filling and interpolating. Remember that it is crucial to choose the adequate interpolation method for the task at hand. Special care needs to be taken when looking at forecasting tasks (for example if you want to use your interpolated data for forecasting weather than you need to remember that you cannot interpolate the weather of today using the weather of tomorrow since it is still unknown).

In order to interpolate the data, we will make use of the *groupby()* function followed by *resample()*. However, first we need to convert the read dates to datetime format and set them as index of our dataframe:


```python
df = df0.copy()
df['datetime'] = pd.to_datetime(df['datetime'])
df.index = df['datetime']
del df['datetime']
```

This is how the structure of the dataframe looks like now:


```python
df.head(1)
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
      <th>house</th>
      <th>readvalue</th>
    </tr>
    <tr>
      <th>datetime</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2018-01-15</th>
      <td>house1</td>
      <td>0.5</td>
    </tr>
  </tbody>
</table>
</div>



## Interpolation 

Since we want to interpolate for each house separately, we need to group our data by 'house' before we can use the *resample()* function with the option 'D' to resample the data to daily frequency. 

The next step is then to use mean-filling, forward-filling or backward-filling to determine how the newly generated grid is supposed to be filled.

### mean()
Since we are strictly upsampling, using the *mean()* method, all missing read values are filled with NaNs:


```python
df.groupby('house').resample('D').mean().head(4)
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
      <th></th>
      <th>readvalue</th>
    </tr>
    <tr>
      <th>house</th>
      <th>datetime</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="4" valign="top">house1</th>
      <th>2018-01-15</th>
      <td>0.500000</td>
    </tr>
    <tr>
      <th>2018-01-16</th>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2018-01-17</th>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2018-01-18</th>
      <td>0.793893</td>
    </tr>
  </tbody>
</table>
</div>



### pad() - forward filling
Using *pad()* instead of *mean()* forward-fills the NaNs.


```python
df_pad = df.groupby('house')\
            .resample('D')\
            .pad()\
            .drop('house', axis=1)
df_pad.head(4)
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
      <th></th>
      <th>readvalue</th>
    </tr>
    <tr>
      <th>house</th>
      <th>datetime</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="4" valign="top">house1</th>
      <th>2018-01-15</th>
      <td>0.500000</td>
    </tr>
    <tr>
      <th>2018-01-16</th>
      <td>0.500000</td>
    </tr>
    <tr>
      <th>2018-01-17</th>
      <td>0.500000</td>
    </tr>
    <tr>
      <th>2018-01-18</th>
      <td>0.793893</td>
    </tr>
  </tbody>
</table>
</div>



### bfill - backward filling
Using *bfill()* instead of *mean()* backward-fills the NaNs:


```python
df_bfill = df.groupby('house')\
            .resample('D')\
            .bfill()\
            .drop('house', axis=1)
df_bfill.head(4)
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
      <th></th>
      <th>readvalue</th>
    </tr>
    <tr>
      <th>house</th>
      <th>datetime</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="4" valign="top">house1</th>
      <th>2018-01-15</th>
      <td>0.500000</td>
    </tr>
    <tr>
      <th>2018-01-16</th>
      <td>0.793893</td>
    </tr>
    <tr>
      <th>2018-01-17</th>
      <td>0.793893</td>
    </tr>
    <tr>
      <th>2018-01-18</th>
      <td>0.793893</td>
    </tr>
  </tbody>
</table>
</div>



### interpolate() - interpolating
If we want to mean interpolate the missing values, we need to do this in two steps. First, we generate the data grid by
using *mean()* to generate NaNs. Afterwards we fill the NaNs by interpolated values by calling the
*interpolate()* method on the readvalue column:


```python
df_interpol = df.groupby('house')\
                .resample('D')\
                .mean()
df_interpol['readvalue'] = df_interpol['readvalue'].interpolate()
df_interpol.head(4)
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
      <th></th>
      <th>readvalue</th>
    </tr>
    <tr>
      <th>house</th>
      <th>datetime</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="4" valign="top">house1</th>
      <th>2018-01-15</th>
      <td>0.500000</td>
    </tr>
    <tr>
      <th>2018-01-16</th>
      <td>0.597964</td>
    </tr>
    <tr>
      <th>2018-01-17</th>
      <td>0.695928</td>
    </tr>
    <tr>
      <th>2018-01-18</th>
      <td>0.793893</td>
    </tr>
  </tbody>
</table>
</div>



## Visualizing the Results
Finally we can visualize the three different filling methods to get a better idea of their results. The opaque dots show the interpolated values.

We can clearly see how in the top figure, the gaps have been filled with the last known value, in the middle figure, the gaps have been filled with the next value to come and in the bottom figure the difference has been interpolated.

<figure class="align-center">
  <img src="{{ site.url }}{{ site.baseurl }}/images/interpolating-timeseries-p1-pandas-fig2.png" alt="">
    <figcaption>Original data (dark) and interpolated data (light), interpolated using (top) forward filling, (middle)
    backward filling and (bottom) interpolation.</figcaption>
</figure>


## Summary

In this blog post we have seen how we can use Python Pandas to interpolate time series data using either backfill, forward fill or interpolation methods. Having used this example to set the scene, in the next post, we will see how to achieve the same thing using PySpark. 
