# Module 10 Challenge - Crypto Clustering

## Overview

In this challenge, I assume the role of a financial advisor proposing a unique approach to structuring investment portfolios based on cryptocurrencies. In pitching this idea, I will integrate unsupervised learning algorithms to go beyond the typical analysis of returns and volatility. 

## Technical Details

The notebook will load the following libraries and dependencies. 

```python
import pandas as pd
import hvplot.pandas
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
```
The data for this project will be imported from a csv file utilizing the Pandas `.read_csv` method and returned into a DataFrame for analysis.  The csv file contains price change percentage data over seven distinct time windows for 41 cryptocurrencies.  The time windows include:

* Price change % 24 hour
* Price change % 7 day
* Price change % 14 day
* Price change % 30 day
* Price change % 60 day
* Price change % 200 day 
* Price change % 1 year

The data will be normalized with the `StandardScaler()` module from `scikit-learn` to prepare it for analysis. It will then be parsed utilizing the `KMeans` model set to the correct number of clusters according to elbow curve data. Each cryptocurrency will be assigned to one of four cluster options by the algorithm and then visualized as a scatter plot utilizing `.hvplot`.

The data will be run through the `KMeans` algorithm once again, this time being optimized with Principal Component Analysis (PCA) to reduce it to three principal components.  The two approaches will be plotted for analysis and compared.  

Despite using fewer features with the PCA approach, the same clusters were achieved, with two distinct outliers clearly visible in both scatter plots, the ethlend and celsius-degree-token cryptocurrencies.

## Sources

The following sources were consulted in the completion of this project. 

* [Holoviews Plotting Documentation](https://hvplot.holoviz.org/)
* [pandas.Pydata.org API Reference](https://pandas.pydata.org/docs/reference/index.html)
* UCB FinTech Bootcamp instructor-led coding exercises
