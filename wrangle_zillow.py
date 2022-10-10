#### Import Section
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import env
from os.path import exists

from itertools import product
from scipy.stats import levene , pearsonr, spearmanr, mannwhitneyu, f_oneway, ttest_ind
from sklearn.metrics import mean_squared_error, explained_variance_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression, TweedieRegressor, LassoLars
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import RFE, f_regression, SelectKBest

import warnings
warnings.filterwarnings("ignore")

# !!!!!!!! WRITE UP A MODULE DESCRIPTION

def get_connection(db, user=env.user, host=env.host, password=env.password):
    ''' 
    takes in database name and import from env
    returns string to use for acquiring info
    '''
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'

def get_zillow_data():
    '''
    Reads in all fields from the customers table in the mall_customers schema from data.codeup.com
    
    parameters: None
    
    returns: a single Pandas DataFrame with the index set to the primary customer_id field
    '''

    sql = """
    SELECT 
        prop.*,
        predictions_2017.logerror as logerror,
        aircon.airconditioningdesc as aircon,
        arch.architecturalstyledesc as architecture,
        buildclass.buildingclassdesc as building_class, 
        heating.heatingorsystemdesc as heating,
        landuse.propertylandusedesc as landuse, 
        story.storydesc as story,
        construct_type.typeconstructiondesc as construct_type
    FROM properties_2017 prop
    JOIN (
        SELECT parcelid, MAX(transactiondate) AS max_transactiondate
        FROM predictions_2017
        GROUP BY parcelid
        ) pred USING (parcelid)
    JOIN predictions_2017 ON pred.parcelid = predictions_2017.parcelid
                        AND pred.max_transactiondate = predictions_2017.transactiondate
    LEFT JOIN propertylandusetype landuse USING (propertylandusetypeid)
    LEFT JOIN airconditioningtype aircon USING (airconditioningtypeid)
    LEFT JOIN architecturalstyletype arch USING (architecturalstyletypeid)
    LEFT JOIN buildingclasstype buildclass USING (buildingclasstypeid)
    LEFT JOIN heatingorsystemtype heating USING (heatingorsystemtypeid)
    LEFT JOIN storytype story USING (storytypeid)
    LEFT JOIN typeconstructiontype construct_type USING (typeconstructiontypeid)
    WHERE propertylandusedesc IN ("Single Family Residential", "Inferred Single Family Residential") 
        AND transactiondate like '%%2017%%';
    """

    if exists('zillow_data.csv'):
        df = pd.read_csv('zillow_data.csv')
    else:
        df = pd.read_sql(sql, get_connection('zillow'))
        df.to_csv('zillow_data.csv', index=False)
    return df

def nulls_by_col(df):
    ''' 
    takes in dataframe
    makes a dataframe of the results from counting nulls in column
    returns that dataframe
    '''
    num_missing = df.isnull().sum()
    percnt_miss = num_missing / df.shape[0] * 100
    cols_missing = pd.DataFrame(
        {
            'num_rows_missing': num_missing,
            'percent_rows_missing': percnt_miss
        }
    )
    return cols_missing

def nulls_by_row(df):
    ''' 
    takes in dataframe
    makes a dataframe of results from counting nulls in row
    returns that dataframe
    '''
    num_missing = df.isnull().sum(axis=1)
    prnt_miss = num_missing / df.shape[1] * 100
    rows_missing = pd.DataFrame({'num_cols_missing': num_missing, 'percent_cols_missing': prnt_miss})
    rows_missing = rows_missing.reset_index().groupby(['num_cols_missing', 'percent_cols_missing']).count().reset_index()

    return rows_missing

def engineer_features(df):
    """
    takes in dataframe (tailored use case for zillow)
    makes an age feature
    renames features
    narrows down values of aircon
    makes openness feature
    makes tax/sqft feature
    uses quantiles to categorize home size
    makes est tax rate feature
    renames county values
    returns df
    """

    #remove unwanted columns, and reset index to id --> for the exercises
    #age
    df['age'] = 2022 - df['yearbuilt']

    #log error bin
    #df['logerror_bin'] = pd.cut(df.logerror,[-6, df.logerror.mean() - df.logerror.std(), 
    #                        df.logerror.mean() + df.logerror.std(), 10],labels=['<-1sig','-1sig~1sig','>1sig'])
    
    #rename 
    df = df.rename(columns={'fips': 'county',
                            'bedroomcnt': 'bedrooms', 
                            'bathroomcnt':'bathrooms', 
                            'calculatedfinishedsquarefeet': 'area', 
                            'taxvaluedollarcnt': 'home_value',
                            'yearbuilt': 'year_built', 
                            'taxamount': 'tax_amount', 
                            })

    #not best way due to adapting from encoding
    df["aircon"] =  np.where(df.aircon == "None", "None",
                                np.where(df.aircon == "Central", "Central", "Other"))

    ## feature to determine how open a house is likely to be
    df["openness"] = df.area / (df.bathrooms + df.bedrooms)
    ## feature to determine the relative tax per sqft
    df["tax_per_sqft"] = df.structuretaxvaluedollarcnt / df.area
    
    # #### Decades: 
    # #create list to hold labels for decades
    # decade_labels = [x + 's' for x in np.arange(1870, 2030, 10)[:-1].astype('str')]

    # #assign decades created from range to new decades column in dataset and apply labels
    # df['decades'] = pd.cut(df.year_built, np.arange(1870, 2030, 10), labels=decade_labels, ordered=True)

    #### Home Size
    #use quantiles to calculate subgroups and assign to new column
    q1, q3 = df.area.quantile([.25, .75])
    df['home_size'] = pd.cut(df.area, [0,q1,q3, df.area.max()], labels=['small', 'medium', 'large'], right=True)

    #### Estimated Tax Rate
    df['est_tax_rate'] = df.tax_amount / df.home_value

    df.county = df.county.map({6037: 'LA County', 6059: 'Orange County', 6111: 'Ventura County'})

    return df

def summarize(df):
    '''  
    takes in dataframe
    prints out df info and df describe as well as the amount of nulls in df by column and row
    then it finds the dtypes of "number" and makes the inverse categorical
    prints out the value counts of cats, as well as binning and doing the same for numerical
    '''
    print('-----')
    print('DataFrame info:\n')
    print (df.info())
    print('---')
    print('DataFrame describe:\n')
    print (df.describe())
    print('---')
    print('DataFrame null value asssessment:\n')
    print('Nulls By Column:', nulls_by_col(df))
    print('----')
    print('Nulls By Row:', nulls_by_row(df))
    numerical_cols = df.select_dtypes(include='number').columns.to_list()
    categorical_cols = df.select_dtypes(exclude='number').columns.to_list()
    print('value_counts: \n')
    for col in df.columns:
        print(f'Column Names: {col}')
        if col in categorical_cols:
            print(df[col].value_counts())
        else:
            print(df[col].value_counts(bins=10, sort=False, dropna=False))
            print('---')
    print('Report Finished')
    return


def handle_missing_values(df, prop_required_columns=0.60, prop_required_row=0.75):
    ''' 
    takes in a dataframe as well as threshould numbers for columns and rows
    drops any column or row that does exceeds threshould
    returns dataframe
    '''
    threshold = int(round(prop_required_columns * len(df.index), 0))
    df = df.dropna(axis=1, thresh=threshold)
    threshold = int(round(prop_required_row * len(df.columns), 0))
    df = df.dropna(axis=0, thresh=threshold)

    return df


def split_data(df):
    ''' 
    takes in dataframe
    uses train test split on data frame using test size of 2, returns train_validate, test
    uses train test split on train_validate using test size of .3, returns train and validate
    returns train, validate test
    '''

    train_validate, test = train_test_split(df, test_size= .2, random_state=514)
    train, validate = train_test_split(train_validate, test_size= .3, random_state=514)
    print(train.shape, validate.shape, test.shape)
    return train, validate, test


def scale_split_data (train, validate, test):
    ''' 
    takes in your three datasets
    applies minmax scaler to them using dtypes of number
    fits to those columns
    applies to copies of datasets
    returns datasets scaled
    '''

    #create scaler object
    scaler = MinMaxScaler()

    # create copies to hold scaled data
    train_scaled = train.copy(deep=True)
    validate_scaled = validate.copy(deep=True)
    test_scaled =  test.copy(deep=True)

    #create list of numeric columns for scaling
    num_cols = train.select_dtypes(include='number')

    #fit to data
    scaler.fit(num_cols)

    # apply
    train_scaled[num_cols.columns] = scaler.transform(train[num_cols.columns])
    validate_scaled[num_cols.columns] =  scaler.transform(validate[num_cols.columns])
    test_scaled[num_cols.columns] =  scaler.transform(test[num_cols.columns])

    return train_scaled, validate_scaled, test_scaled


def prep_zillow (df):
    """ 
    Purpose
        Perform preparation functions on the zillow dataset
    Parameters
        df: data acquired from zillow dataset
    Output
        df: the unsplit and unscaled data with removed columns
    """
    
    df = engineer_features(df)

    df = df.drop(columns=['parcelid', 'buildingqualitytypeid','censustractandblock', 'calculatedbathnbr',
                        'heatingorsystemtypeid', 'propertylandusetypeid', 'year_built', 
                        'rawcensustractandblock', 'landuse', 'fullbathcnt', 'finishedsquarefeet12',
                        'assessmentyear', 'regionidcounty','regionidzip', 'regionidcity','tax_amount',
                        'propertycountylandusecode', 'propertyzoningdesc', 'roomcnt'])
    df = df.set_index('id')

    #fill na values
    df.heating.fillna('None', inplace=True)
    df.aircon.fillna('None', inplace=True)
    df.basementsqft.fillna(0,inplace=True)
    df.garagecarcnt.fillna(0,inplace=True)
    df.garagetotalsqft.fillna(0,inplace=True)
    df.unitcnt.fillna(1,inplace=True)
    df.poolcnt.fillna(0, inplace=True)

    df = df[ (df["bedrooms"] > 0) | (df["bathrooms"] > 0) ]

    #not best way due to adapting from encoding
    df["heating"] =  np.where(df.heating == "Central", "Central",
                                np.where(df.heating == "None", "None",
                                np.where(df.heating == "Floor/Wall", "Floor/Wall", "Other")))

    # handle the missing data --> decisions made in advance
    df = handle_missing_values(df, prop_required_columns=0.64)

    # take care of unitcnts (more than 1 and was not nan when brought in)
    df = df[df["unitcnt"] == 1]

    # take care of any duplicates:
    df = df.drop_duplicates()

    #drop na/duplicates --> adjust this for project. think of columns to impute
    df = df.dropna()

    df.drop(columns="unitcnt",inplace=True)
    
    #split the data
    train, validate, test = split_data(df)

    dict_to_cluster = { "house_tax":["tax_per_sqft","est_tax_rate","openness"],
                    "house_details":["lotsizesquarefeet",'garagetotalsqft',"poolcnt"],
                    "house_sizing":['area', 'bathrooms', 'bedrooms'],
                    "house_locale":["latitude","longitude","age"],
                    "cluster_order":[5,4,5,5]}

    train,validate,test = get_kmeans_cluster_features(train,validate,test,dict_to_cluster)

    #scale the data
    train_scaled, validate_scaled, test_scaled = scale_split_data(train, validate, test)

    return df, train, validate, test, train_scaled, validate_scaled, test_scaled  

def get_kmeans_cluster_features(train,test,validate,dict_to_cluster):
    ''' 
    takes in your three datasets to apply the new featuers to as well as
    a dictionary (iterable lists with the last being the order of clustering (function to auto later))
    '''

    for i in range(len(list(dict_to_cluster))-1):
        #set features
        X1 = train[dict_to_cluster[list(dict_to_cluster)[i]]]
        X2 = validate[dict_to_cluster[list(dict_to_cluster)[i]]]
        X3 = test[dict_to_cluster[list(dict_to_cluster)[i]]]

        # make, fit, apply:
        scaler = MinMaxScaler()
        # train is all numerical so I can call the whole thing
        X1_scaled = pd.DataFrame(scaler.fit_transform(X1),index = X1.index,columns = X1.columns)
        X2_scaled = pd.DataFrame(scaler.transform(X2),index = X2.index,columns = X2.columns)
        X3_scaled = pd.DataFrame(scaler.transform(X3),index = X3.index,columns = X3.columns)

        kmeans_scaled = KMeans(n_clusters=dict_to_cluster[list(dict_to_cluster)[len(list(dict_to_cluster))-1]][i],random_state=123)
        kmeans_scaled.fit(X1_scaled)

        X1_scaled["cluster"] = kmeans_scaled.predict(X1_scaled)
        X2_scaled["cluster"] = kmeans_scaled.predict(X2_scaled)
        X3_scaled["cluster"] = kmeans_scaled.predict(X3_scaled)

        train[f"cluster {list(dict_to_cluster)[i]}"] = X1_scaled["cluster"]
        validate[f"cluster {list(dict_to_cluster)[i]}"] = X2_scaled["cluster"]
        test[f"cluster {list(dict_to_cluster)[i]}"] = X3_scaled["cluster"]

    return train,test,validate

def wrangle_zillow():
    """ 
    Purpose
        Perform acuire and preparation functions on the zillow dataset
    Parameters
        None
    Output
        df: the unsplit and unscaled data
        X_train:
        X_train_scaled:
        X_validate:
        X_validate_scaled:
        X_test:
        X_test_scaled:
    """
    #initial data acquisition
    df = get_zillow_data()
    
    #drop columns that are unneeded, split data
    df, train, validate, test, train_scaled, validate_scaled, test_scaled = prep_zillow(df)

    #summarize the data --> commented out for final presentation
    #summarize(df)

    return df, train, validate, test, train_scaled, validate_scaled, test_scaled
