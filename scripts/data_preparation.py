import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams["figure.figsize"] = (20,10)

df1 = pd.read_csv("Bengaluru_House_Data.csv")
print(df1.head())

# Shape will print(no of rows and no of columns)
print(df1.shape)

#this line is combining the all different values is in area type
# then i have use agg ( aggregate function) to count the how many different
# values are there
agg = df1.groupby('area_type')['area_type'].agg("count")

print("Aggregate funciton: \n",agg)
print(agg.sum())

# Now droping values which are not important for predicting price

df2 = df1.drop(['area_type','society','balcony','availability'],axis = 'columns')
print(df2.head())

# now we have to check for null values:
# like how many null values are there:
null = df2.isnull().sum()
print("Null Values")
print(null)
print("Total ",null.sum())

# now our data set is large that why we are simply
# droping the na values:
df3 = df2.dropna()
print(df3.head())

# now we have different value of size (ex: bhk, bedrooms ..e.t.c:)
# so, what we have to, we have to transform these values and pick 
# only the first element of size:

unique = df3['size'].unique()
print("Unique values in size: ",unique)

# now apply transform funciton:

df3["bhk"] = df3['size'].apply(lambda x: int(x.split(" ")[0]))
print("added bhk in the end \n",df3.head())

print("printing the unique value in total sqft: ",df3.total_sqft.unique())
#in output we are getting range in total_sqft so we have to take the average:

# now we are check for any float values:

def is_float(x):
    try:
        float(x)
    except:
        return False
    return True

# now we have to tackle with the range values:
# in this i have used negate operator (~)
checker = df3[~df3['total_sqft'].apply(is_float)].head(10)
print(checker)

# in this total_sqft we are getting in ( 34.46Sq. Meter ),   (4125Perch)
# for this you have to two option first remove these values or use sqft function to convertt
# these value into sqft

# now writing funciton to give value by giving average of range values in total sqft

def convert_sqrt_to_num(x):
    tokens = x.split('-')
    if len(tokens) == 2:
        return (float(tokens[0])+float(tokens[1]))/2
    try:
        return float(x)
    except:
        return None
    
    
df4 = df3.copy()
df4['total_sqft'] = df4['total_sqft'].apply(convert_sqrt_to_num)
print("\n",df4.head(3))
print("\n ",df4.loc[648])

#video 2: Feature Engineering


print(df4.isnull().sum())
df5 = df4.dropna()
print(df5.isnull().sum())
print(df5.head())

df6 = df5.copy()
# our price is 1 lacks rupees that why we write 100000
df6['price_per_sqft'] = df6['price']*100000/df5['total_sqft']
print(df6.head())

# checking how many location do we have
print(len(df6.location.unique()))
# now we have 1298 different places so, for that we have to check how many 
# different cityies with less then 10 posts:
df6.location = df6.location.apply(lambda x: x.strip())
location_stats = df6.groupby('location')['location'].agg('count').sort_values(ascending = False)
print(location_stats)

# now we are finding values which are less then equal  to 10
print(len(location_stats[location_stats <=10]))
locaiton_stats_less_then_10 = location_stats[location_stats <= 10]
print("Length: ",locaiton_stats_less_then_10)
print("Total: ",len(location_stats[location_stats<=533]))


print(len(df6.location.unique()))
df6.location = df6.location.apply(lambda x: 'other' if x in locaiton_stats_less_then_10 else x)
print(len(df6.location.unique()))

print(df6.head())

# video 3 Outliear Detection
# we can use two techniques for outliear detection: 
# 1) standard deviation 2) simple domin knowledge

# find data error (outliears) 
outlinear = df6[df6.total_sqft/df6.bhk<300].head()
print(outlinear)
print(df6.shape)

# Removing (outliears) error values:
df7 = df6[~(df6.total_sqft/df6.bhk<300)]
print(df7.shape)

# now we are done with the outlinear in the total sqft

# now we moving to price:
# basic statitics info of price per sqft

print(df7.price_per_sqft.describe())
# now from this we have to find error values (outlinears) from this

# outliner removal function
def remove_pps_outliers(df):
    df_out = pd.DataFrame()
    for key, subdf in df.groupby('location'):
        m = np.mean(subdf.price_per_sqft)
        st = np.std(subdf.price_per_sqft)
        reduced_df = subdf[(subdf.price_per_sqft >(m-st)) & (subdf.price_per_sqft <=(m+st))]
        df_out = pd.concat([df_out,reduced_df],ignore_index = True)
    return df_out

df8 = remove_pps_outliers(df7)
print(df8.shape)

# ploting 2bhk and  3bhk pricing like 2 bhk have higher price then 3 bhk
# for that we are ploting

def plot_scatter_chart(df,location):
    bhk2 = df[(df.location == location) & (df.bhk == 2)]
    bhk3 = df[(df.location == location) & (df.bhk == 3)]
    matplotlib.rcParams['figure.figsize'] = (15,10)
    plt.scatter(bhk2.total_sqft,bhk2.price,color = 'blue',label = '2 BHK',s = 50)
    plt.scatter(bhk3.total_sqft,bhk3.price,marker = "+",color = 'green',label = '3 BHK',s = 50)
    plt.xlabel("Total Square Feet Area")
    plt.ylabel("Price")
    plt.title(location)
    plt.legend()
    plt.show()
    
#plot_scatter_chart(df8,"Rajaji Nagar")
#plot_scatter_chart(df8,"Hebbal")

# now again we have to do Data Cleaning

#we should also remove properties where same location , the price of (for example)
# 3 bedroom apartments is less than 2 bedroom apartments(with same square ft area)
# what we will do is dfor a given location, we will build a dictionary of stats per bhk
# i.e

# {
#         '1' : {
#             'mean': 4000,
#             'std': 2000,
#             'count': 34
#         },
#         '2' : {
#             'mean': 4300,
#             'std': 2300,
#             'count': 22
#         },
# }

# now we can remove those 2 bhk aparments whose price_per_sqft is less 
# then mean price_per_sqft of 1 BHK aparment

def remove_bhk_outliers(df):
    exclude_indices = np.array([])
    for location, location_df in df.groupby('location'):
        bhk_stats = {}
        for bhk, bhk_df in location_df.groupby('bhk'):
            bhk_stats[bhk] = {
                'mean': np.mean(bhk_df.price_per_sqft),
                'std' : np.std(bhk_df.price_per_sqft),
                'count': bhk_df.shape[0]
            }
            
        for bhk,bhk_df in location_df.groupby('bhk'):
            stats = bhk_stats.get(bhk-1)
            if stats and stats['count']>5:
                exclude_indices = np.append(exclude_indices, bhk_df[bhk_df.price_per_sqft<(stats['mean'])].index.values)
    return df.drop(exclude_indices,axis = "index")

df9 = remove_bhk_outliers(df8)
print(df9.shape)
#plot_scatter_chart(df9,"Hebbal")

# now ploting histogram
# matplotlib.rcParams["figure.figsize"] = (10,5)
# plt.hist(df9.price_per_sqft,rwidth = 0.8)
# plt.xlabel("Price per Square Feet")
# plt.ylabel("Count")
#plt.show()

# now we targeting BathRoom Variable

print(df8.bath.unique())
print(df9[df9.bath>10])

#now creating the histogram for bathrooms

# plt.hist(df9.bath, rwidth=0.8)
# plt.xlabel("Number of Bathrooms")
# plt.ylabel("Count")
# plt.title("Distribution of Bathrooms")
# plt.show()


df10 = df9[df9.bath<=df9.bhk+2]
print("Outlier 2 in this bath" ,df10)
print("Num of rows and columns: ",df10.shape)

# now we dont need size, price per sqft so, we will remove these

df11 = df10.drop(['size','price_per_sqft'],axis = 'columns')
print(df11.head(3))

# video 5 in this we will be using machine learning models
#  we will create machine Model Building

# now we creating dummies variable for location:
print("Printing dummies")
dummies = pd.get_dummies(df11.location)
print(dummies.head(3))

df12 = pd.concat([df11,dummies.drop('other',axis = 'columns')],axis = 'columns')

print(df12.head())

df13 = df12.drop('location',axis = 'columns')
print(df13.head(3))

# these df,1 ,2,3,4,5,... 13 are called piplines

# now we are building the model for our data
print(df13.shape)

X = df13.drop('price',axis = 'columns')
y = df13.price


# now we will use train test split method:
from sklearn.model_selection import train_test_split
# what is randam_state = 10?
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state = 10)

from sklearn.linear_model import LinearRegression
lr_clf = LinearRegression()
lr_clf.fit(X_train,y_train)



print(lr_clf.score(X_test,y_test))

import pickle
import json
import os

# Path to save files
model_save_path = './app/artifacts/banglore_home_prices_model.pickle'
columns_save_path = './app/artifacts/columns.json'

# Create the folder if it doesn't exist
os.makedirs('./app/artifacts', exist_ok=True)

# Save the trained model
with open(model_save_path, 'wb') as f:
    pickle.dump(lr_clf, f)
print(f"✅ Model saved to: {model_save_path}")

# Save the data columns used in the model
columns = {
    'data_columns': [col.lower() for col in X.columns]  # Lowercase to match input
}
with open(columns_save_path, 'w') as f:
    json.dump(columns, f)
print(f"✅ Columns saved to: {columns_save_path}")


# here we are using cross validation
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
cv = ShuffleSplit(n_splits = 5, test_size =0.2, random_state = 0)
print(cross_val_score(LinearRegression(),X,y,cv = cv))

print('code perfect till here')

# now we will use grid search cv

from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeRegressor

#creating a function which we choose the best model
def find_best_model_using_gridsearch(X, y):
    algos = {
        'linear_regression': {
            'model': LinearRegression(),
            'params': {
                'fit_intercept': [True, False],  # Changed from normalize
                'positive': [True, False]  
            }
        },
        'lasso': {
            'model': Lasso(),
            'params': {
                'alpha': [1, 2],
                'selection': ['random', 'cyclic']
            }
        },
        'decision_tree': {
            'model': DecisionTreeRegressor(),
            'params': {
                'criterion': ['friedman_mse', 'squared_error'],
                'splitter': ['best', 'random']
            }
        }
    }

    scores = []
    cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
    for algo_name, config in algos.items():
        gs = GridSearchCV(config['model'], config['params'], cv=cv, return_train_score=False)
        gs.fit(X, y)
        scores.append({
            'model': algo_name,
            'best_score': gs.best_score_,
            'best_params': gs.best_params_
        })

    return pd.DataFrame(scores, columns=['model', 'best_score', 'best_params'])

best_models_df = find_best_model_using_gridsearch(X, y)
print(best_models_df)
print(X.columns)
loc = np.where(X.columns == '2nd Phase Judicial Layout')[0][0]
print(loc)

def predict_price(location,sqft,bath,bhk):
    loc_index = np.where(X.columns == location)[0][0]
    
    x = np.zeros(len(X.columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if loc_index >=0:
        x[loc_index] = 1
        
    return lr_clf.predict([x])[0]


price1 = predict_price('1st Phase JP Nagar',1000,2,2)
print(price1)
price2 = predict_price('Indira Nagar',1000,3,3)
print(price2)

# now we are dont with the model building:
# now we will use joblib:
# after that we do flask work

#now we are importing this project
import pickle
with open('banglore_home_prices_model.pickle','wb') as f:
    pickle.dump(lr_clf,f)

import json
columns = {
    'data_columns' : [col.lower() for col in X.columns]
    
}
with open('columns.json','w') as f:
    f.write(json.dumps(columns))