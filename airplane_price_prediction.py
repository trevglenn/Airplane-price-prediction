import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('Airplane_price_prediction\Plane Price.csv')

print(df.head())
df.info()

""" 
[5 rows x 16 columns]
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 517 entries, 0 to 516
Data columns (total 16 columns):
 #   Column                   Non-Null Count  Dtype
---  ------                   --------------  -----
 0   Model Name               517 non-null    object
 1   Engine Type              517 non-null    object
 2   HP or lbs thr ea engine  517 non-null    object
 3   Max speed Knots          497 non-null    float64
 4   Rcmnd cruise Knots       507 non-null    float64
 5   Stall Knots dirty        502 non-null    float64
 6   Fuel gal/lbs             517 non-null    int64
 7   All eng rate of climb    513 non-null    object
 8   Eng out rate of climb    491 non-null    float64
 9   Takeoff over 50ft        492 non-null    float64
 10  Landing over 50ft        516 non-null    object
 11  Empty weight lbs         516 non-null    object
 12  Length ft/in             516 non-null    object
 13  Wing span ft/in          516 non-null    object
 14  Range N.M.               499 non-null    object
 15  Price                    507 non-null    float64
dtypes: float64(6), int64(1), object(9)
memory usage: 64.8+ KB 

"""

# --------------------------------------------------------------------------

""" Need to manually correct our dtypes for numerical and categorical data. Also, will manually fill our null values. """

# Starting with Length and Wingspan: For these two columns we need to separate our recorded feet and inches values, divide inches by 12, then add to feet value. 

df['Length ft/in'] = df['Length ft/in'].astype('string')
df['Length ft/in'] = df['Length ft/in'].fillna('0/0')
df['Length ft/in'] = df['Length ft/in'].replace('N/C', '0/0', regex=True)
df['Length ft/in'] = df['Length ft/in'].replace('/', ',', regex=True)

length = df['Length ft/in']
#print(df['Length ft/in'].head())
true_length = []

for i in range(len(length)):
    length_split = []
    length_split = length[i].split(',')
    length_ft = float(length_split[0])
    length_in = float(length_split[1])
    total_length = length_ft + (length_in / 12)
    true_length.append(total_length)

df['Length ft/in'] = true_length
#print(df['Length ft/in'])

# Wingspan is missing values for inches that are not recorded as Null because the feet measurement was still recorded.
# We will be dropping the inches values and just moving forward with feet for Wingspan. 

df['Wing span ft/in'] = df['Wing span ft/in'].astype('string')
df['Wing span ft/in'] = df['Wing span ft/in'].fillna('0/0')
df['Wing span ft/in'] = df['Wing span ft/in'].replace('N/C', '0/0', regex=True)
df['Wing span ft/in'] = df['Wing span ft/in'].replace('/', ',', regex=True)

wing_span = df['Wing span ft/in']
#print(wing_span.head())
true_span = []

for i in range(len(wing_span)):
    w_split = []
    w_split = wing_span[i].split(',')
    length_ft = float(w_split[0])
    true_span.append(length_ft)

df['Wing span ft/in'] = true_span
#print(df['Wing span ft/in'])

# -----------------------------------------------------------------

""" 
Still need to manually fix: 
1. HP or lbs thr ea engine
2. All eng rate of climb
3. Landing over 50ft
4. Empty weight lbs
5. Range N.M.
6. Engine Type

"""
#print(df['Engine Type'].nunique()) -- 4 Unique Engine Types should be good for ohe

# 1. HP or lbs thr ea engine needs to be converted to float

df['HP or lbs thr ea engine'] = df['HP or lbs thr ea engine'].replace(',', '', regex=True)
df['HP or lbs thr ea engine'] = df['HP or lbs thr ea engine'].astype('float')
df['HP or lbs thr ea engine'] = df['HP or lbs thr ea engine'].fillna(df['HP or lbs thr ea engine'].mean()) 

# 2. All eng rate of climb needs to be converted to float

df['All eng rate of climb'] = df['All eng rate of climb'].replace(',', '', regex=True)
df['All eng rate of climb'] = df['All eng rate of climb'].astype('float')
df['All eng rate of climb'] = df['All eng rate of climb'].fillna(df['All eng rate of climb'].mean()) 

# 3. Landing over 50 ft needs to be converted to float

df['Landing over 50ft'] = df['All eng rate of climb'].replace(',', '',regex=True)
df['Landing over 50ft'] = df['All eng rate of climb'].astype('float')
df['Landing over 50ft'] = df['Landing over 50ft'].fillna(df['Landing over 50ft'].mean()) 

# 4. Empty weight lbs needs to be converted to float

df['Empty weight lbs'] = df['Empty weight lbs'].replace(',', '', regex=True)
df['Empty weight lbs'] = df['Empty weight lbs'].astype('float')
df['Empty weight lbs'] = df['Empty weight lbs'].fillna(df['Empty weight lbs'].mean()) 

# 5. Range N.M. needs to be converted to float

df['Range N.M.'] = df['Range N.M.'].replace(',', '', regex=True)
df['Range N.M.'] = df['Range N.M.'].astype('float')
df['Range N.M.'] = df['Range N.M.'].fillna(df['Range N.M.'].mean()) 

#6. Engine Type needs to be converted to string

df['Engine Type'] = df['Engine Type'].astype('string')

#7. Filling our remaining Null values
""" Max speed Knots, Rcmnd cruise knots, Stall Knots dirty, Eng out rate of climb, Takeoff over 50ft. """

df['Max speed Knots'] = df['Max speed Knots'].fillna(df['Max speed Knots'].mean())
df['Rcmnd cruise Knots'] = df['Rcmnd cruise Knots'].fillna(df['Max speed Knots'].mean())
df['Stall Knots dirty'] = df['Stall Knots dirty'].fillna(df['Stall Knots dirty'].mean())
df['Eng out rate of climb'] = df['Eng out rate of climb'].fillna(df['Eng out rate of climb'].mean())
df['Takeoff over 50ft'] = df['Takeoff over 50ft'].fillna(df['Takeoff over 50ft'].mean())

## Converting model name to string, then drop it from our dataframe as it's an unnecessary categorical column with too many unique values
df['Model Name'] = df['Model Name'].astype('string')
df = df.drop(columns=['Model Name'])


## Converting Fuel gal/lbs to same dtype as rest of dataset
df['Fuel gal/lbs'] = df['Fuel gal/lbs'].astype('float')

# -----------------------------------------------------------------

df['Price'] = df['Price'].fillna(0)

y = df['Price']
for i in range(len(y)):
    if y[i] == 0:
        y[i] = y.mean()

X = df.iloc[:, 0:-1]

#df.info()

"""
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 517 entries, 0 to 516
Data columns (total 15 columns):
 #   Column                   Non-Null Count  Dtype
---  ------                   --------------  -----
 0   Engine Type              517 non-null    string
 1   HP or lbs thr ea engine  517 non-null    float64
 2   Max speed Knots          517 non-null    float64
 3   Rcmnd cruise Knots       517 non-null    float64
 4   Stall Knots dirty        517 non-null    float64
 5   Fuel gal/lbs             517 non-null    float64
 6   All eng rate of climb    517 non-null    float64
 7   Eng out rate of climb    517 non-null    float64
 8   Takeoff over 50ft        517 non-null    float64
 9   Landing over 50ft        517 non-null    float64
 10  Empty weight lbs         517 non-null    float64
 11  Length ft/in             517 non-null    float64
 12  Wing span ft/in          517 non-null    float64
 13  Range N.M.               517 non-null    float64
 14  Price                    517 non-null    float64
dtypes: float64(14), string(1)

"""

# -----------------------------------------------------------------


cat_cols = X['Engine Type']
print(cat_cols)
cat_cols = pd.get_dummies(cat_cols)

num_cols = X.drop(columns=['Engine Type'])
print(num_cols)

sc = StandardScaler()
sc.fit_transform(num_cols)

X = pd.concat([cat_cols, num_cols], axis=1)
print(X)


# ------------------------------------------------------------------


x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)


# ------------------------------------------------------------------

## Linear Regression model

lr = LinearRegression()

lr.fit(x_train, y_train)
y_pred = lr.predict(x_test)
score = lr.score(x_test, y_test)

print(f"Our linear regression model score is: {score}")
### Our linear regression model score is 0.859288011485787


# Ridge Regression model

ridge = Ridge()

ridge.fit(x_train, y_train)
ridge_pred = ridge.predict(x_test)
ridge_score = ridge.score(x_test, y_test)
print(f'Our Ridge Regression score is: {ridge_score}')
### Our ridge regression model score is 0.8601160652709332


# Lasso Regression model

lasso = Lasso(alpha=0.1)

lasso.fit(x_train, y_train)
lasso_pred = lasso.predict(x_test)
lasso_score = lasso.score(x_test, y_test)
print(f'Our Lasso score is: {lasso_score}')

### Our Lasso Regression score is 0.8592882050535107

# --------------------------------------------------------------------

plt.scatter(y_test, y_pred, alpha=0.4)
plt.title('Features to Price - Linear Regression Model')
plt.xlabel('True Price ($ Million)')
plt.ylabel('Predicted Price ($ Million)')
plt.show()
plt.clf()


plt.scatter(y_test, ridge_pred, alpha=0.4)
plt.title('Features to Price - Ridge Regression Model')
plt.xlabel('True Price ($ Million)')
plt.ylabel('Predicted Price ($ Million)')
plt.show()
plt.clf()


plt.scatter(y_test, lasso_pred, alpha=0.4)
plt.title('Features to Price - Lasso Regression Model')
plt.xlabel('True Price ($ Million)')
plt.ylabel('Predicted Price ($ Million)')
plt.show()
plt.clf()

# -----------------------------------------------------------------


"""
Our best ML regression model so far has been the Ridge Model, which works best when all features have relatively the same importance.
Our basic LR model had a similar score to Lasso, with Lasso only having a very slightly higher score. 
This tells us that our feature importance is likely relatively balanced, so Ridge would be the best model for the data if we were looking to tune our hyperparameters.

"""