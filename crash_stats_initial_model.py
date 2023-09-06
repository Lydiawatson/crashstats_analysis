import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt

from sklearn import ensemble
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.utils import resample
from sklearn.inspection import permutation_importance


#set random seed for repeatability
RANDOM_STATE = 2023
np.random.seed(RANDOM_STATE)

#helper functions
def convert_key_df(value_pairs, source_df):
    """
    Uses numeric_code and string descriptor columns in source_df to create a multi-index df key for translating numeric codes
    
    """
    if isinstance(value_pairs, tuple):
        value_pairs = [value_pairs]
       
    #for each field, get the  value/descriptor mapping
    key_dfs = []
    for val, desc in value_pairs:
        df = source_df[[val, desc]].drop_duplicates().reset_index(drop=True).rename(columns={val: 'Value', desc: 'Desc'})
        df['Field_Type'] = val
        key_dfs.append(df)


    #merge to a multi-index df
    key_df = pd.concat(key_dfs, axis=0, ignore_index=True)
    key_df.set_index(['Field_Type', 'Value'], inplace=True)    
    
    return key_df


#calculate metrics
def calc_precision(pred, ytrue, value=1):
    """ Calculate precision for class 'value'"""
    actual_positives =  ytrue.to_numpy()[np.where(pred == value)]
    true_positives = np.sum(actual_positives == value)
    
    return true_positives/len(actual_positives)

#recall
def calc_recall(pred, ytrue, value=1):
    """ Calculate recall for class 'value'"""
    pred_positives = pred[np.where(ytrue.to_numpy() == value)]
    true_positives = np.sum(pred_positives == value)
    
    return true_positives/len(pred_positives)


#-------------------------------------------
#----------------
# Data Cleaning and feature extraction
#----------------
#-------------------------------------------

#----------------
# Base 'accident' table
#----------------
subfolder = 'ACCIDENT'

#load and clean main accident info
acc_df = pd.read_csv(f'{subfolder}/ACCIDENT.csv')


#clean date/time features
#rename date and timecolumns to match naming convention
acc_df.rename({'ACCIDENTDATE': 'ACCIDENT_DATE', 
              'ACCIDENTTIME': 'ACCIDENT_TIME',
              }, axis=1, inplace=True)

#convert date to datetime
acc_df.ACCIDENT_DATE = pd.to_datetime(acc_df.ACCIDENT_DATE, yearfirst=True)

acc_df['HOUR_OF_DAY'] = acc_df['ACCIDENT_TIME'].apply(lambda x: x.split(':')[0]).astype('int')
acc_df['MONTH_OF_YEAR'] = acc_df['ACCIDENT_DATE'].apply(lambda x: x.month)


#recalculate existing day-of-week column, where Monday is 1, Sunday is 7
#(In the original dataframe, the day of week and day week description are not consistent (e.g. Friday corresponds to 6,5,2 at different times))
acc_df['DAY_OF_WEEK'] = acc_df['ACCIDENT_DATE'].apply(lambda x: x.isoweekday())
acc_df.drop(labels='Day Week Description', axis=1, inplace=True)



#recalculate severity based on no. of people injured columns (as initial database is not consistent)
def severity_calc(row):

    if row['NO_PERSONS_KILLED'] > 0:
        return 1
    elif row['NO_PERSONS_INJ_2'] > 0:
        return 2
    elif row['NO_PERSONS_INJ_3'] > 0:
        return 3
    elif row['NO_PERSONS_NOT_INJ'] > 0:
        return 4
    else:
        return 9
    
acc_df['SEVERITY'] = acc_df.apply(severity_calc, axis = 1)


#separate str descriptors of numeric codes to a separate df
value_desc_pairs = [('ACCIDENT_TYPE','Accident Type Desc'),
                    ('DCA_CODE', 'DCA Description'), 
                    ('LIGHT_CONDITION','Light Condition Desc'),
                    ('ROAD_GEOMETRY','Road Geometry Desc')]

key_df = convert_key_df(value_desc_pairs, source_df = acc_df)


#drop descriptor columns and unwanted map columns
acc_df = acc_df.drop(['Accident Type Desc',
                     'DCA Description',
                     'DIRECTORY',
                     'EDITION',
                     'PAGE',
                     'GRID_REFERENCE_X',
                     'GRID_REFERENCE_Y',
                     'Light Condition Desc',
                     'Road Geometry Desc',
                     'ACCIDENT_TIME'], axis=1)

#----------------
# Reading and merging other tables
#----------------

#atmospheric condition table describes weather (e.g. rainy/cloudy)
df = pd.read_csv(f'{subfolder}/ATMOSPHERIC_COND.csv')

#add atmospheric conditions to key df
key_df = pd.concat([key_df, convert_key_df(('ATMOSPH_COND','Atmosph Cond Desc'), source_df = df)], axis=0)
key_df.tail(10)

# add atmospheric condition to main df
acc_df = acc_df.merge(df[['ACCIDENT_NO','ATMOSPH_COND']], how='left', on='ACCIDENT_NO')


#Node table contains the urbanisation descriptor (e.g. Melbourne Urban vs small town)
df = pd.read_csv(f'{subfolder}/NODE.csv')
df = df[['NODE_ID','DEG_URBAN_NAME']].drop_duplicates(subset='NODE_ID')

acc_df = acc_df.merge(df, how='left', left_on='NODE_ID',right_on='NODE_ID')



#People data: ignoring age of passengers (only selecting drivers) and randomly selecting one driver as 'contributor' -- poor assumption but best available given data
#keep sex and age of the driver, calculate boolean feature based on whether their licence is in-state or not, whether driver was wearing seatbelt

df = pd.read_csv(f'{subfolder}/PERSON.csv')
#exclude all persons not drivers
df = df.loc[df['ROAD_USER_TYPE'] == 2].copy()

#create boolean features for instate licence/seatbelt
df['INSTATE_LICENCE'] = df['LICENCE_STATE'] == 'V'
df['HELMET_BELT_WORN'] = df['HELMET_BELT_WORN'].replace({' ':9}).astype('int')
df['SEATBELT_WORN'] = df['HELMET_BELT_WORN'] == 1 #unknown or other protection treated as no

df = df[['ACCIDENT_NO','VEHICLE_ID','SEX','AGE','INSTATE_LICENCE','SEATBELT_WORN']]

#select only one driver from each event (flawed assumption)
df = df.sample(n=len(df), replace=False, random_state=RANDOM_STATE).drop_duplicates(subset='ACCIDENT_NO').copy()

#merge driver details
acc_df = acc_df.merge(df, on='ACCIDENT_NO',how='outer')
acc_df.sort_values('ACCIDENT_DATE', inplace=True)


#vehicle data, want age of vehicle, total no occupants
df = pd.read_csv(f'{subfolder}/VEHICLE.csv')
df = df[['ACCIDENT_NO','VEHICLE_ID','VEHICLE_YEAR_MANUF','TOTAL_NO_OCCUPANTS']]
acc_df = acc_df.merge(df, on=['ACCIDENT_NO','VEHICLE_ID'], how='inner')

#----------------
# Cleaning - remove events with null/unknown values
#----------------
print(f"df shape before cleaning {acc_df.shape}")
acc_df = acc_df.drop(['DCA_CODE','NODE_ID','POLICE_ATTEND','VEHICLE_ID'],axis=1)

#convert crash-stats 'null' codes to nans/NA
acc_df['ROAD_GEOMETRY'] = acc_df['ROAD_GEOMETRY'].replace({9: np.nan})
acc_df['LIGHT_CONDITION'] = acc_df['LIGHT_CONDITION'].replace({9: np.nan})
acc_df['SPEED_ZONE'] = acc_df['SPEED_ZONE'].replace({777:np.nan, 888:np.nan, 999:np.nan})
acc_df['ATMOSPH_COND'] = acc_df['ATMOSPH_COND'].replace({9: np.nan})
acc_df['SEX'] = acc_df['SEX'].replace({' ': pd.NA, 'U': pd.NA})
acc_df['VEHICLE_YEAR_MANUF'] = acc_df['VEHICLE_YEAR_MANUF'].replace({0:pd.NA, 3001:pd.NA})
acc_df['TOTAL_NO_OCCUPANTS'] = acc_df['TOTAL_NO_OCCUPANTS'].replace({0:pd.NA})

#drop all events with null values
acc_df = acc_df.dropna(axis=0)


#remove events in 2020 in case of lockdown effects
acc_df = acc_df.loc[acc_df.ACCIDENT_DATE < np.datetime64('2020')].copy()

print(f"df shape after cleaning {acc_df.shape}")

print("Column names:")
print(acc_df.columns.values)



#-------------------------------------------
#----------------
# Model development
#----------------
#-------------------------------------------

#----------------
# Prepare dataset for training
#----------------

#convert string dtypes to numeric
X = pd.concat([acc_df, pd.get_dummies(acc_df['DEG_URBAN_NAME'])], axis=1)
X['SEX'] = X['SEX'].replace({'M':1, 'F':0})

#remove non-feature columns
X = X.drop(['SEVERITY','NO_PERSONS_INJ_2', 'NO_PERSONS_INJ_3', 'NO_PERSONS_KILLED','NO_PERSONS_NOT_INJ', 'DEG_URBAN_NAME', 'ACCIDENT_NO','ACCIDENT_DATE'], axis=1)
y = acc_df['SEVERITY']

#split into training/testing datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, shuffle=True,  random_state=RANDOM_STATE)

#----------------
# Train  model and evaluate
#----------------
#initialise random forest
rf = ensemble.RandomForestClassifier(n_estimators = 50, random_state = RANDOM_STATE, criterion="entropy")
rf.fit(X_train, y_train)


y_predict = rf.predict(X_test)

print(f"Overall accuracy: {rf.score(X_test, y_test)}")

for class_val in [1,2,3]:
    print(f"Precision for {class_val}: \t {calc_precision(y_predict, y_test, value=class_val)}")
    print(f"Recall for {class_val}: \t\t {calc_recall(y_predict, y_test, value=class_val)}\n")
    





#----------------
# Visualise Feature Importance
#----------------

results = permutation_importance(rf, X_test, y_test, n_jobs=-1, random_state=RANDOM_STATE)

#plot results
fig, ax = plt.subplots()

perm_sorted_idx = results.importances_mean.argsort()
feature_names = X.columns

num_plot=15

ax.boxplot(
    results.importances[perm_sorted_idx].T,
    vert=False,
labels=feature_names) #[:15])

ax.set_title('Feature Importance using Permutation (test set)')
ax.set_xlabel('Decrease in accuracy')

ax.axvline(x=0, color="k", linestyle="--")
plt.show()




