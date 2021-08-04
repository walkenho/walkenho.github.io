---
title: >-
    We Are Beating Child Mortality - How to Create Interactive and Animated Visualizations in Python 
header:
  overlay_image: /images/beergarden-happiness-overlay.jpg
  overlay_filter: 0.5 # same as adding an opacity of 0.5 to a black background
  caption: "Photo by [**Tomasz Rynkiewicz on Unsplash**](https://unsplash.com/@thmsr)"
  actions:
    - label: "Enjoying a Drink Outside"
classes: wide
tags:
  - python
  - visualization
  - open data
excerpt: Recreating Hans Rosling's famous bubble charts in Python.
---


In 2009 Hans Rosling -founder of [Gapminder](https://www.gapminder.org/)- gave a [TED talk](https://www.ted.com/talks/hans_rosling_the_good_news_of_the_decade_we_re_winning_the_war_against_child_mortality) about the world-wide evolution of child mortality since the 1960s. He came to the conclusion that at that time, we were on a good way to beating child mortality, even though we were not quite there yet. He evidenced his statements by showing the evolution of child mortality vs female fertility using his famous bubble graphs. 

Being the amazing storyteller, he is, Rosling's talk left me wondering where we were on our journey in 2019, 10 years later. So I decided to figure out the next chapter in the story, revise where we were with respect to child mortality and hone my visualization skills by learning how to do animated graphs in Python. This post describes my findings. As always, the full notebook is available on [my GitHub](https://github.com/walkenho/tales-of-1001-data/tree/master/child-mortality).

Let's dig-in. First, we need to import the modules.

## Loading the Modules


```python
import time
import wget
import pandas_flavor as pf

import numpy as np
import pandas as pd

# bokeh modules
from bokeh.io import (output_notebook,
                      push_notebook, 
                      show)
from bokeh.models import (ColumnDataSource,
                          HoverTool,
                          Label)
from bokeh.plotting import figure
from bokeh.models.glyphs import Text

# interactive widgets
from ipywidgets import interact

output_notebook()
```



    


## Obtaining and Loading the Data
First of all, we need to source and load the data. We are interested in visualizing the interplay of child mortality and fertility on a country level. This data can be obtained from the [World Bank Open Data Initiative](https://data.worldbank.org/), where we find [data about child mortality](https://data.worldbank.org/indicator/SH.DYN.MORT) and [data about fertility](https://data.worldbank.org/indicator/SP.DYN.TFRT.IN). Keeping in Rosling's tradition we use a bubble graph as visualization type. To add a bit more flavour to our figures, we encode [data about a country's population](https://data.worldbank.org/indicator/SP.POP.TOTL) as bubble size and color it by its region.

A quick inspection of the source files shows that the first three rows of the excel files are headers and should be dropped when loading the files. We also note that the tables contain both data for individual countries and aggregated numbers. We are only interested in the data on individual country level. In order to filter out the aggregated entries, we can use the second sheet (Metadata - Countries). For each of the data files, the metadata sheet contains the country codes and the associated regions. Looking closer, we see that the regions are null for aggregated data. This means that if we filter the country codes for non-null regions, we will keep a list of only the non-aggregated, individual country values, which we can then use to filter our data using an in inner join. 

In order to drop the aggregates, we use the `pandas_flavor` library, which allows us to transform a function into a dataframe method.

Finally we extract the maximum year for which the sheets contain data. This will be used for the animation. It can be obtained from the column headings. Note that the final column is empty, so we need to take the maximum column heading minus one as maximum year.

Let's do it.


```python
@pf.register_dataframe_method
def drop_aggregates(df, myfilter):
    """Keep only rows whose index exists in the index of myfilter.
    
    Return:
    df1 - dataframe
    """
    
    df1 = df.join(myfilter, how="inner")
    return df1


def read_datafile(filename):
    """Load relevant columns of file into dataframe.
    
    Return:
    df - dataframe
    """
    
    df = (
        pd.read_excel(filename, header=3, index_col='Country Code')
        .drop(['Country Name', 'Indicator Name', 'Indicator Code'], axis=1)
    )
    return df

# create filter for nulls and load populations
filename = wget.download("http://api.worldbank.org/v2/en/indicator/SP.POP.TOTL?downloadformat=excel")
countryfilter = (
    pd.read_excel(filename,
                  sheet_name='Metadata - Countries',
                  index_col="Country Code")
    .loc[:,[('Region')]]
    .dropna()
)

df_population = (
    read_datafile(filename)
    .drop_aggregates(myfilter=countryfilter)
)

# fertility
filename = wget.download("http://api.worldbank.org/v2/en/indicator/SP.DYN.TFRT.IN?downloadformat=excel")
#filename = "API_SP.DYN.TFRT.IN_DS2_en_excel_v2_41097.xls"
df_fertility = (
    read_datafile(filename)
    .drop_aggregates(myfilter=countryfilter)
)

#mortality
filename = wget.download("http://api.worldbank.org/v2/en/indicator/SH.DYN.MORT?downloadformat=excel")
#filename = "API_SH.DYN.MORT_DS2_en_excel_v2_180823.xls"
df_mortality = (
    read_datafile(filename)
    .drop_aggregates(myfilter=countryfilter)
)


def find_min_and_max_year(dfs):
    """Return maximum populated year from list of dataframes
    
    Return:
    max_year - int
    """
    
    min_years = []
    max_years = []
    for df in dfs:
        years = []
        for c in df.columns:
            try:
                years.append(int(c))
            except ValueError:
                pass
        min_years.append(min(years))
        max_years.append(max(years) - 1)
    min_year = max(min_years)
    max_year = min(max_years)
    
    return min_year, max_year
```

We will use the geographical region of a country to colour the bubbles and also label each one of them with its country name. Let's create the data sets.


```python
# download and load the countrynames required for labeling
countrynames = (
    pd.read_excel(filename,
                  sheet_name='Metadata - Countries',
                  index_col="Country Code")
    .loc[:,[('TableName')]]
    .drop_aggregates(myfilter=countryfilter)
)

# create colour list to be used for plotting
colour_dict = {'Latin America & Caribbean':'green', 
               'South Asia' : 'red', 
               'Sub-Saharan Africa' : 'yellow',
               'Europe & Central Asia':'blue', 
               'Middle East & North Africa': 'orange', 
               'East Asia & Pacific': 'violet',  
               'North America':'darkgreen'}
c = [colour_dict[region] for region in countryfilter['Region']]
```

Now that we have sourced the data, we are going to explore how we can create both an animation as well as an interactive graph on top of it. There are multiple libraries out there to achieve this. Here, we will look at [`Bokeh`](https://docs.bokeh.org/en/latest/), a library that provides a python wrapper around the JavaScript library [d3.js](https://d3js.org/). When creating a Bokeh visualization, the user creates a base figure and attaches a handle. This handle can then be used (implicitly or explicitly) to update the figure in-situ. We can update figure properties as well as the underlying data.

## Creating the Base Figure
To make it easier to update the underlying data, we first create a function that takes a year as input and creates a Bokeh ColumnDataSource containing all the relevant data for this year. This ColumnDataSource can then be used to update the Bokeh figure either interactively through the use of a user-operated widget or dynamically through a loop creating an animation.

We also use the function to create the base figure for the first year in our dataset.


```python
def create_dataset(year):
    """Create ColumnDataSource containing data for year.
    
    Parameters:
    year - integer
    
    Returns:
    src - ColmnDataSource
    """
    
    data = {'fertility': df_fertility[str(year)],
        'mortality' : df_mortality[str(year)]/10,
        'population': df_population[str(year)].map(lambda x: "{:3.1f} Mio".format(x/1e7)),
        'size': df_population[str(year)].map(lambda x: np.sqrt(x)/10**3*5),
        'countrynames' : countrynames['TableName'],
        'colourcode' : c}
    df = pd.DataFrame(data, 
                      columns = ['fertility', 'mortality',
                                 'population', 'size',
                                 'countrynames', 'colourcode'])
    src = ColumnDataSource(df)
    return src

# find first year in dataset
year_start, _ = find_min_and_max_year([df_population, df_fertility, df_mortality])

# create starting data set
src = create_dataset(year_start)
```

Next we define the figure. However, since the figure gets updated in place and we would like it to be at the end of the notebook, we do not display it quite yet :)


```python
# Create the figure
p = figure(title="Child Mortality vs Fertility", 
           plot_height=400,
           plot_width=800, 
           x_range=(1,9),
           y_range=(-5,40),
           x_axis_label='Fertility [# of children per woman]',
           y_axis_label='Child Mortality [%]',
           background_fill_color='#efefef')

# Add data from source src
r = p.scatter(source=src, 
              x='fertility', 
              y='mortality', 
              size='size', 
              color='colourcode', 
              line_width=2, 
              line_color='grey',
              alpha=0.5)

# Add background text showing the year.
glyph_src = ColumnDataSource(dict(x=[4.8], y=[35], text=[str(year_start)]))
glyph = Text(x='x', y='y', text='text')
p.add_glyph(glyph_src, glyph)

# Add title.
p.title.text = "Development of Child Mortality vs Fertility by Country"

# Add hovertips.
hover = HoverTool(tooltips = [('Country', '@countrynames'), ('Population', '@population')])
p.add_tools(hover)
```

We can now create a function to update the plot. We need to update both the data and the glyph containing the background text. After updating the data sources for figure and glyph, the changes need to be pushed into the figure. This is done (in this case implicitly) using a handle, which is defined when creating the figure.


```python
def update(year):
    """Update src and glyph_src with the data for year."""
    
    # update data
    new_src = create_dataset(year)
    src.data.update(new_src.data)
    
    # update glyph
    new_glyph_src = ColumnDataSource(dict(x=[4.8], y=[35], text=[str(year)])).data
    glyph_src.data.update(new_glyph_src)
    
    # if we wanted to add the year to the title, this is how to update it
    # p.title.text = "Child Mortality vs Fertility " + str(year)
    
    # push changes to the notebook 
    # push_notebook pushes to last handle
    # alternatively handle could be specified like this:
    # push_notebook(handle=myhandle)
    push_notebook()
```

We now have a base figure and a function to update the data sources for the figure. These are the basics for the our animated/interactive graphs. We can now either create an animated figure or a figure with a widget that allows the user to interact with it.

## Animating a Graph
In order to create an animation, we need to define start and end year as well as how often we want the animation to cycle through the time span. In the following code, the animation runs for all years with data in the dataset and it completes the cycle three times. In order to slow down the animation speed, I added a small `sleep` step at each iteration and a larger `sleep` step at the beginning and the end of the animation, so the user realizes that it is starting again.


```python
# Minimum and maximum year of animation
year_start, year_end = find_min_and_max_year([df_population, df_fertility, df_mortality])
n_years = year_end - year_start

# Number of times animation runs.
n_rounds = 2

# Calculate total number of frames to be used as counter.
total_number_of_frames = n_years*n_rounds

# Display figure, creating handle 
show(p, notebook_handle=True)

# Movie time
for _ in range(n_rounds):
    for year in range(year_start, year_end+1):
        update(year)

        # create delay between frames
        # make delay longer for first and last frame
        if (year == year_start):
            time.sleep(0.5)
        elif (year == year_end):
            time.sleep(1)
        else:
            time.sleep(0.1)
```

<iframe src="/images/beating-child-mortality-bokeh.html"
    sandbox="allow-same-origin allow-scripts"
    width="100%"
    height="500"
    scrolling="yes"
    seamless="seamless"
    frameborder="0">
</iframe>

## Creating an Interactive Graph
The second option I wanted to explore was how to create interactive graphs. This can be achieved with the help of the `interact` method. `interact()` takes a function as argument and allows the user to change its input parameters using a widget. In the following, we allow the user to change the input parameter year of the function update between the limits 1960 and 2017.


```python
# Display figure, creating handle 
show(p, notebook_handle=True)
_ = interact(update, year=(year_start,year_end))
```













    interactive(children=(IntSlider(value=1989, description='year', max=2018, min=1960), Output()), _dom_classes=(…


## Summary and Conclusions
In this post, I have shown how we can leverage python in combination with bokeh to easily generate animations and interactive graphs. We have used this to revisit the state of child mortality at the end of 2019 - 10 years after Rosling concluded that we were on a good way to beating child mortality. The updated data shows that even though we are still not there, the situation keeps on improving year upon year and now child mortality is almost consistently under 10% :)
