#### Import Section
from lib2to3.refactor import get_all_fix_names
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
import requests
import os

from itertools import product

from sklearn.metrics import mean_squared_error, explained_variance_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression, TweedieRegressor, LassoLars
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import RFE, f_regression, SelectKBest

def get_opsd_germany():
    ''' 
    no inputs
    attempts to find relevant csv, returns df if found, if not it attempts to get it
    '''

    if os.path.isfile("opsd_germany.csv"):
        opsd_germany_df = pd.read_csv('opsd_germany.csv')
    else:
        print("can't find opsd_germany, will attempt to get")
        url = "https://raw.githubusercontent.com/jenfly/opsd/master/opsd_germany_daily.csv"
        opsd_germany_df = pd.read_csv(url)
        opsd_germany_df.to_csv('opsd_germany.csv', index=False)

    return opsd_germany_df

def get_zguide():
    ''' 
    no input
    looks for csv, if not found then it attempts to run functions and concats. creates csv of merged
    '''
    if os.path.isfile("zguide_sales_combined.csv"):
        df = pd.read_csv('zguide_sales_combined.csv')
    else:
        #runs functions
        sales_df = get_zguide_sales()
        items_df = get_zguide_items()
        stores_df = get_zguide_stores()
        #concats results
        df = pd.merge(sales_df,stores_df,how="left",left_on="store",right_on="store_id")
        df = pd.merge(df,items_df,how="left",left_on="item",right_on="item_id")
        df.to_csv('zguide_sales_combined.csv', index=False)
    return df

def get_zguide_stores():
    ''' 
    no input
    looks for csv, does get if cant find
    '''
    if os.path.isfile("zguide_stores.csv"):
        stores_df = pd.read_csv('zguide_stores.csv')
    else:
        response = requests.get('https://python.zgulde.net/api/v1/stores')
        data = response.json()

        current_page = data['payload']['page']
        max_page = data['payload']['max_page']
        next_page = data['payload']['next_page']

        print(f'current_page: {current_page}')
        print(f'max_page: {max_page}')
        print(f'next_page: {next_page}')

        stores_df = pd.DataFrame(data['payload']['stores'])
        stores_df.to_csv('zguide_stores.csv', index=False)

    return stores_df


def get_zguide_items():
    ''' 
    not input
    looks for csv, does get if cant find
    '''
    if os.path.isfile("zguide_items.csv"):
        items_df = pd.read_csv('zguide_items.csv')
    else:
        print("can't find zguide items")
        #sets the info
        base_url = 'https://python.zgulde.net'
        response = requests.get('https://python.zgulde.net/api/v1/items')
        data = response.json()

        current_page = data['payload']['page']
        next_page = data['payload']['next_page'][-1:]
        max_page = data['payload']['max_page']
        print(f'current_page: {current_page}')
        print(f'next_page: {next_page}')
        print(f'max_page: {max_page}')

        #first page set
        items_df = pd.DataFrame(data['payload']['items'])

        #get and concat the rest of the pages
        for i in range(int(next_page),int(max_page)+1):
            response = requests.get(base_url + data['payload']['next_page'])
            data = response.json()
            current_page = data['payload']['page']
            print(f'current_page: {current_page}')
            items_df = pd.concat([items_df, pd.DataFrame(data['payload']['items'])])

        items_df.to_csv('zguide_items.csv', index=False)

    return items_df

def get_zguide_sales():
    '''
    looks for csv if not found attempts to get
    '''
    # looks for file, if not found goes to get it
    if os.path.isfile("zguide_sales.csv"):
        sales_df = pd.read_csv('zguide_sales.csv')
    else:
        print("can't find zguide sales")
        #sets the info
        base_url = 'https://python.zgulde.net'
        response = requests.get('https://python.zgulde.net/api/v1/sales')
        data = response.json()

        current_page = data['payload']['page']
        next_page = data['payload']['next_page'][-1:]
        max_page = data['payload']['max_page']
        print(f'current_page: {current_page}')
        print(f'next_page: {next_page}')
        print(f'max_page: {max_page}')

        #first page set
        sales_df = pd.DataFrame(data['payload']['sales'])
        #get and concat the rest of the pages
        for i in range(int(next_page),int(max_page)+1):

            response = requests.get(base_url + data['payload']['next_page'])
            data = response.json()

            current_page = data['payload']['page']

            print(f'current_page: {current_page}')

            sales_df = pd.concat([sales_df, pd.DataFrame(data['payload']['sales'])])

        sales_df.to_csv('zguide_sales.csv', index=False)

    return sales_df