#### Import Section
from lib2to3.refactor import get_all_fix_names
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
import requests
import acquire
import os

from itertools import product

from sklearn.metrics import mean_squared_error, explained_variance_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression, TweedieRegressor, LassoLars
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import RFE, f_regression, SelectKBest

def prep_sales():
    ''' 
    no inputs due to desiring direct reproduction
    takes the dataframe from acquiring sales and does:
        drop unused columns
        plots distributions for quick review
        turns the sale_date into datetime format, and makes it the index
        creates month and day of week from datetime
        new feature of sales total
    '''

    #acquires the data and drops features with duplicated inof
    df = acquire.get_zguide()
    df.drop(columns=["sale_id","store_id","item_id","item","item_upc12","item_upc14"],inplace=True)

    df.shape

    print("store ids =", df.store.unique(),
      "| store cities =", df.store_city.unique(),
      "| store states =", df.store_state.unique())

    print("sale amount count =",df.sale_amount.nunique(),
          "| sale date count =",df.sale_date.nunique())
    
    print("item brand count =", df.item_brand.nunique(),
      "| item name count =", df.item_name.nunique(),
      "| item_price count =", df.item_price.nunique())

    #plots some distributions for review
    plt.hist(x=df["sale_amount"])
    plt.xlabel("sale amount")
    plt.ylabel("count")
    plt.title("Distribution of Sale amount")
    plt.show()

    plt.hist(x=df["item_price"].astype(int))
    plt.xlabel("item price")
    plt.ylabel("count")
    plt.title("Distribution of item price")
    plt.show()

    #changes data to datetime and creates features from it
    df.sale_date = df.sale_date.str.replace("00:00:00 GMT","")
    df.sale_date = df.sale_date.str.strip()
    df.sale_date = pd.to_datetime(df.sale_date,format="%a, %d %b %Y")
    df = df.set_index("sale_date").sort_index()    
    df["month"] = df.index.month_name()
    df["day_of_week"] = df.index.day_name()
    df["sales_total"] = df["sale_amount"] * df["item_price"]

    return df
    
def prep_germany_opsd():
    ''' 
    no inputs due to desiring direct reproduction
    takes the dataframe from acquiring german opsd and does:
        plots distributions for quick review
        turns the sale_date into datetime format, and makes it the index
        creates month and day of week from datetime
        new feature of sales total
    '''
    #get data
    df = acquire.get_opsd_germany()

    #send date to index after turning into datetime pandas format
    df.Date = pd.to_datetime(df.Date)
    df = df.set_index("Date").sort_index()

    #plots distribution of all features
    plt.figure(figsize=(20, 10))
    for i, col in enumerate(df.columns.tolist()): # List of columns
        plot_number = i + 1 # i starts at 0, but plot nos should start at 1
        plt.subplot(2,int(len(df.columns.tolist())/2), plot_number) # Create subplot.
        plt.xlabel(col)
        plt.ylabel("count")
        plt.title(f"Count of {col}") # Title with column name.
        df[col].hist(bins=10) # Display histogram for column.
        plt.grid(False) # Hide gridlines.

    #feature engineers and imputes 0 for NaN/Null
    df["month"] = df.index.month_name()
    df["year"] = df.index.year
    print("is null before\n",df.isnull().sum())
    df.fillna(0,inplace=True)
    print("is null after\n",df.isnull().sum())
    
    return df