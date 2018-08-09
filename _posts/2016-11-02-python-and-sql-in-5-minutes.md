---
title: Python and SQL in 5 Minutes
tags:
  - python
  - pandas
  - sql
excerpt: An introduction to combining Python with SQL
---

Recently I played around with a combination of SQL and Python. I had been quite interested in learning SQL before and
had been working through Stanford's excellent collection of
[self-paced database courses](https://lagunita.stanford.edu/courses/DB/2014/SelfPaced/about). Since I normally
work in Python, I was looking for an way of combining SQL and Python. And I found a dream team! While
using SQL in combination with Python can sometimes seem a bit cumbersome, combining SQL with the Pandas package is
easier than anything. In fact, I found that one can achieve a reasonable lot with only eight commands.

Here is a summary of the main functionality.

## Step 0 - Preparing the Data

Let's assume that we start with three csv-files that include information about movie ratings: movies.csv,
reviewers.csv and ratings.csv (this is in example database from [Stanford's SQL
course](https://lagunita.stanford.edu/courses/DB/SQL/SelfPaced/about). 

First, we have to import the necessary modules. Here, I will be working with SQLite3.


```python
import pandas as pd
import sqlite3
```

Next, we use the pandas read_csv command to read in the csv-files:


```python
movies = pd.read_csv('movies.csv')
reviewers = pd.read_csv('reviewers.csv')
ratings = pd.read_csv('ratings.csv')
```

We can inspect the dataframes by:


```python
movies.head(2)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>mID</th>
      <th>title</th>
      <th>year</th>
      <th>director</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>101</td>
      <td>Gone with the Wind</td>
      <td>1939</td>
      <td>Victor Fleming</td>
    </tr>
    <tr>
      <th>1</th>
      <td>102</td>
      <td>Star Wars</td>
      <td>1977</td>
      <td>George Lucas</td>
    </tr>
  </tbody>
</table>
</div>



## Step 1 - Creating the Database

First, we need to create a connection:


```python
conn = sqlite3.connect("rating.db")
```

We can use the connection to directly save the dataframes as tables of our database. 

df.to_sql() has the option "index" which is by default set to True. This saves the index column of the dataframe in the table. Since I am not interested in saving the df index, I set index=False:


```python
reviewers.to_sql("reviewers", conn, if_exists="replace", index=False)
ratings.to_sql("ratings", conn, if_exists="replace", index=False)
movies.to_sql("movies", conn, if_exists="replace", index=False)
```

## Step 3: Querying

In order to query our database, we use the pd.read_sql_query command, which directly returns a dataframe which can then be analyzed with the usual dataframe options:


```python
pd.read_sql_query("""select * from reviewers""", conn).head(2)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>rID</th>
      <th>name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>201</td>
      <td>Sarah Martinez</td>
    </tr>
    <tr>
      <th>1</th>
      <td>202</td>
      <td>Daniel Lewis</td>
    </tr>
  </tbody>
</table>
</div>



In other databases, the same could be achieved with "pd.read_sql_table("reviewers", conn)", but Sqlite's DBAPI is not supported here.

Note that sqlite3 does not natively support date as datatype. When querying, we can tell it to treat columns as datetype by setting the "parse_dates" option. Note the difference between columns ratingDate and date:   


```python
pd.read_sql_query("""select rID, mID, ratingDate, ratingDate as date from ratings where stars=4""", conn, parse_dates=["ratingDate"])
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>rID</th>
      <th>mID</th>
      <th>ratingDate</th>
      <th>date</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>201</td>
      <td>101</td>
      <td>2011-01-27</td>
      <td>2011-01-27</td>
    </tr>
    <tr>
      <th>1</th>
      <td>202</td>
      <td>106</td>
      <td>NaT</td>
      <td>None</td>
    </tr>
    <tr>
      <th>2</th>
      <td>203</td>
      <td>108</td>
      <td>2011-01-12</td>
      <td>2011-01-12</td>
    </tr>
    <tr>
      <th>3</th>
      <td>205</td>
      <td>108</td>
      <td>NaT</td>
      <td>None</td>
    </tr>
  </tbody>
</table>
</div>



## Step 4: Changing and Updating Tables

Changing and updating of databases happens in three steps:
1. Define a cursor: **cur = conn.cursor()**
2. Execute command: **cur.execute(SQL-command)**
3. (Commit the change to the db: **conn.commit()**)

### 1. Define a cursor 
In order to update a database, first we have to define a cursor on the connection:


```python
cur = conn.cursor()
```

### 2. Execute a command

#### a) Adding a column: 
For example, we can add a column "NRatings" to the table reviewers (that would be a horrible choice from a design point, but let's go for it)


```python
cur.execute("alter table reviewers add column NRatings integer;")
```




    <sqlite3.Cursor at 0xae8f4520>



If we check reviewers, we see that this took effect immediately:


```python
pd.read_sql_query("""select * from reviewers""", conn).head(2)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>rID</th>
      <th>name</th>
      <th>NRatings</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>201</td>
      <td>Sarah Martinez</td>
      <td>None</td>
    </tr>
    <tr>
      <th>1</th>
      <td>202</td>
      <td>Daniel Lewis</td>
      <td>None</td>
    </tr>
  </tbody>
</table>
</div>



#### b) Updating entries: 
We now populate the created column.


```python
cur.execute("""update reviewers set NRatings = (select count(*) from ratings where ratings.rID = reviewers.rID)""")
```




    <sqlite3.Cursor at 0xae8f4520>



If we check, we see the change:


```python
pd.read_sql_query("""select * from reviewers""", conn).head(2)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>rID</th>
      <th>name</th>
      <th>NRatings</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>201</td>
      <td>Sarah Martinez</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>202</td>
      <td>Daniel Lewis</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



**BUT**: The change is not in the database yet. We can see this by creating a second connection to the same database and use the second connection for querying:


```python
conn2 = sqlite3.connect("rating.db")
```


```python
pd.read_sql_query("""select * from reviewers""", conn2).head(2)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>rID</th>
      <th>name</th>
      <th>NRatings</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>201</td>
      <td>Sarah Martinez</td>
      <td>None</td>
    </tr>
    <tr>
      <th>1</th>
      <td>202</td>
      <td>Daniel Lewis</td>
      <td>None</td>
    </tr>
  </tbody>
</table>
</div>



In order for the update to take effect, we still have to commit it. We do this by typing:


```python
conn.commit()
```

And querying now shows that the update took effect:


```python
pd.read_sql_query("""select * from reviewers""", conn2).head(2)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>rID</th>
      <th>name</th>
      <th>NRatings</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>201</td>
      <td>Sarah Martinez</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>202</td>
      <td>Daniel Lewis</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



This might look a bit strange at first glance, but has the reason to prevent inconsistent databases if the something goes wrong in the process. Imagine the case that money is transfered from one bank account to another. If something goes wrong after the money is subtracted from the first account but before arriving at the second you end up with inconsistent tables. To prevent this you first do both changes then commit them together.

### 3. Word of Caution! 
Here is a word of caution taken from the <a href="https://docs.python.org/2/library/sqlite3.html"> python documention
on sqlite3 </a> because a) security is important and b) it gives me a good reason to link to the xkcd comic below, which made me laugh so much :D

Usually your SQL operations will need to use values from Python variables. You shouldn’t assemble your query using
Python’s string operations because doing so is insecure; it makes your program vulnerable to an SQL injection
attack (see [this](https://xkcd.com/327/) fabulous XKCD for a humorous example of what can go wrong). 
Instead, use the DB-API’s parameter substitution. Put ? as a placeholder wherever you want to use a value, and then provide a tuple of values as the second argument to the cursor’s execute() method.


```python
# Never do this -- insecure!
# title = 'Star Wars'
# cur.execute("""update movies set director='Darth Vader' where title = '%s'""" % title)

# Do this instead
t = ('Star Wars',)
cur.execute("""update movies set director='Darth Vader' where title = ?""", t)
```




    <sqlite3.Cursor at 0xae8f4520>



Or for multiple values:


```python
t = ('Darth Vader', 'Star Wars',)
cur.execute("""update movies set director=? where title = ?""", t)
```




    <sqlite3.Cursor at 0xae8f4520>



### 4. Closing the Connection
Finally, we can close the connection to the database by:


```python
conn.close()

# And we have to close the second one, too:
conn2.close()
```

And that's it folks! This should give you a quick start into combining Pandas/Python with SQL(ite).

I also found <a href="https://www.dataquest.io/blog/python-pandas-databases/"> this tutorial </a> quite helpful, where you can also find more follow-up links. 

Enjoy!
