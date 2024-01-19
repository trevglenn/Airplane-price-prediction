import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_pinball_loss, mean_squared_error

from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

# Ensemble: We will be using GradientBoostingRegressor to build an ensemble method for our dataset.

df = pd.read_csv('Airplane_price_prediction\Plane Price.csv')

print(df.head())
df.info()

#-----------------------------------------------------
# Import our manually cleaned data

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

# 1. HP or lbs thr ea engine needs to be converted to float

df['HP or lbs thr ea engine'] = df['HP or lbs thr ea engine'].replace(',', '', regex=True)
df['HP or lbs thr ea engine'] = df['HP or lbs thr ea engine'].astype('float')
df['HP or lbs thr ea engine'] = df['HP or lbs thr ea engine'].fillna(df['HP or lbs thr ea engine'].mean()) 

# 2. All eng rate of climb needs to be converted to float

df['All eng rate of climb'] = df['All eng rate of climb'].replace(',', '', regex=True)
df['All eng rate of climb'] = df['All eng rate of climb'].astype('float')
df['All eng rate of climb'] = df['All eng rate of climb'].fillna(df['All eng rate of climb'].mean()) 

# 3. Landing over 50 ft needs to be converted to float

df['Landing over 50ft'] = df['Landing over 50ft'].replace(',', '',regex=True)
df['Landing over 50ft'] = df['Landing over 50ft'].astype('float')
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
df['Engine Type'] = df['Engine Type'].replace('piston', 'Piston', regex=True)


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


# -----------------------------------------------------------------


cat_cols = X['Engine Type']
#print(cat_cols)
cat_cols = pd.get_dummies(cat_cols)

num_cols = X.drop(columns=['Engine Type'])
#print(num_cols)

sc = StandardScaler()
sc.fit_transform(num_cols)

X = pd.concat([cat_cols, num_cols], axis=1)
#print(X)


# ------------------------------------------------------------------

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# ------------------------------------------------------------------

params = {
    "n_estimators": 600,
    "max_depth": 4,
    "min_samples_split": 5,
    "learning_rate": 0.01,
    "warm_start": True,
}


ridge = Ridge()
ridge.fit(x_train, y_train)

reg = GradientBoostingRegressor(loss='squared_error',**params)
reg.fit(x_train, y_train)
score = reg.score(x_test, y_test)
print(f"Our Gradient Boosting score is: {score}")
# Our Gradient Boosting score is: 0.8967842214998942
mse = mean_squared_error(y_test, reg.predict(x_test))
print("The mean squared error (MSE) on test set: {:.4f}".format(mse))
# The mean squared error on the test set is 110,257,843,529.92

""" Our model is now close to a 90% accuracy! Let's plot our training and test deviance, then take a look at some of the important features of our output. """


test_score = np.zeros((params["n_estimators"],), dtype=np.float64)
for i, y_pred in enumerate(reg.staged_predict(x_test)):
    test_score[i] = mean_squared_error(y_test, y_pred)

fig = plt.figure(figsize=(6, 6))
plt.subplot(1, 1, 1)
plt.title("Deviance")
plt.plot(
    np.arange(params["n_estimators"]) + 1,
    reg.train_score_,
    "b-",
    label="Training Set Deviance",
)
plt.plot(
    np.arange(params["n_estimators"]) + 1, test_score, "r-", label="Test Set Deviance"
)
plt.legend(loc="upper right")
plt.xlabel("Boosting Iterations")
plt.ylabel("Deviance")
fig.tight_layout()
plt.show()
plt.clf()

# ------------------------------------------------------------

feature_importance = reg.feature_importances_
sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0]) + 0.5
fig = plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.barh(pos, feature_importance[sorted_idx], align="center")
plt.yticks(pos, np.array(X.columns)[sorted_idx])
plt.title("Feature Importance (Price)")

# ------------------------------------------------------------

result = permutation_importance(
    reg, x_test, y_test, n_repeats=10, random_state=42, n_jobs=2
)
sorted_idx = result.importances_mean.argsort()
plt.subplot(1, 2, 2)
plt.boxplot(
    result.importances[sorted_idx].T,
    vert=False,
    labels=np.array(X.columns)[sorted_idx],
)
plt.title("Permutation Importance (test set)")
fig.tight_layout()
plt.show()

# -----------------------------------------------------------

""" 
Most important features in our dataset for our target are:

1. Rcmnd cruise Knots
2. Length ft/in
3. HP or lbs thr ea engine - pounds of thrust the engine has
4. Jet Engine
5. Max Speed knots

"""

# -----------------------------------------------------------

