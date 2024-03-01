# Module 6 Challenge - Housing Rental Analysis for San Francisco

## Overview

This challenge will engage the Proptech world in conducting an analysis of real estate data for San Francisco from 2010 to 2016. The goal is to present a trial version of this software that will aid customers in selecting properties to purchase and present to the market as rentals in the San Francisco area.  

## Technical Details

The notebook will load the following libraries and dependencies.

```python
import pandas as pd
import hvplot.pandas
from pathlib import Path
```

The data for this project will be imported from three distinct csv files utilizing the Pandas `.read_csv` method and returned into a DataFrame for analysis. The data will be visualized as bar, line, line with a group pull-down, and scatter plots, utilizing Holoviews Plotting `hvplot`. 

The first visualization will examine the number of housing units per year from 2010 to 2016 in San Francisco utilizing a `.mean()` of a dataframe that has been grouped by the "year" field. Conclusions will be drawn from this visualization pertaining to overall housing unit trends. This visualization will illustrate the overall growth in number of housing units, year-to-year in San Francisco.

The second visualization will graph the data for both the sale price per square foot and the gross rent for San Francisco over the same period as a plot with two lines. It will validate the idea that gross rent is steadily growing while home sales prices per square foot remain rather flat over the subject period.

The third visualization will plot the same data, this time with "neighborhoods" grouped (`groupby = 'neighborhood'`), creating a pull-down to show data for the selected neighborhood.

The fourth visualization, an interactive neighborhood map (scatter plot), will be a concatenation of two DataFrames, joining the geospatial data to the existing real estate data. The dots plotted will be sized to the `sale_price_sqr_foot` column and the color parameter will be set to `gross_rent`.

Relevant conclusions to the San Francisco real estate market will be drawn by hovering over the subject plots and analyzing the hover data.

## Sources

The following sources were consulted in the completion of this project. 

* [Holoviews Plotting Documentation](https://hvplot.holoviz.org/)
* [pandas.Pydata.org API Reference](https://pandas.pydata.org/docs/reference/index.html)
* UCB FinTech Bootcamp instructor-led coding exercises