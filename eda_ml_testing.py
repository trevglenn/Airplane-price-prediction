import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import Ridge, Lasso
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA

df = pd.read_csv('Airplane_price_prediction\Plane Price.csv')


print(df.head())
df.info()

"""
Notes:
ft/in is a recording of both ft and inches not a division of ft by inches. 
Will need to separate at / symbol, divide inches by 12, then add back to feet to get to a proper float.


"""

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

## Mission accomplished ! Now to do the same to Wing span - additional issue here is we are missing inches values in some of our data. Becuase
## we still have feet recorded in the same column the null value is not recorded


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
Still need to fix: 
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

## Converting model name to string
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

""" 
Need to separate our categorical and numerical variables, will likely need to drop our [Plane] Model Name column. 
Will be using one-hot encoding for our categorical variables and then StandardScale() for our Numerial variables.
Once data is processed we can assemble our first model. 

"""
cat_cols = X['Engine Type']
print(cat_cols)
cat_cols = pd.get_dummies(cat_cols)

num_cols = X.drop(columns=['Engine Type'])
print(num_cols)

sc = StandardScaler()
sc.fit_transform(num_cols)

X = pd.concat([cat_cols, num_cols], axis=1)
print(X)

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

"""
Testing Pipeline: 

df['Price'] = df['Price'].fillna(0)

y = df['Price']
for i in range(len(y)):
    if y[i] == 0:
        y[i] = y.mean()

X = df.iloc[:, 0:-1]

num_cols = X.drop(columns=['Engine Type'])
num_cols = num_cols.replace(',', '', regex=True)
num_cols = num_cols.astype('float')

cat_cols = pd.get_dummies(X['Engine Type'])

print('Columns with missing values:')
print(X.columns[X.isnull().sum()>0])

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)


cat_vals = Pipeline([("imputer", SimpleImputer(strategy='most_frequent')), ("ohe", OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore'))])

num_vals = Pipeline([("imputer", SimpleImputer(strategy='mean')), ("scale", StandardScaler())])

preprocess = ColumnTransformer(transformers=[("cat_process", cat_vals, cat_cols), ("num_process", num_vals, num_cols)])

"""

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


## Very similar scatterplots. 
# -----------------------------------------------------------------

"""
Our best ML regression model so far has been the Ridge Model, which works best when all features have relatively the same importance.
Our basic LR model had a similar score to Lasso with Lasso only having the slightest of a higher score, this tells us that our feature importance is relatively balanced. 

May try an ensemble method next to see if we can apply multiple regression functions to our data to get a much more accurate prediction. 

Then we will build a Deep Learning Regression model to see if that is more or less accurate than our ML models.

"""