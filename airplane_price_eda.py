import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('Airplane_price_prediction\Plane Price.csv')

print(df.head())
df.info()
print(df.columns)

""" Labels for our model will be price. It will be a regression model in which we will predict the price given our features.
Our features consist of both categorical and numerical variables so we will need to preprocess these separately. Our only int64 
feature is Fuel gal/lbs. We now want to take a look at our features and determine which seem to have a strong correlation to our price. """

print(df.isnull().sum())

## 517 samples. We have some null values including some in price that we will need to replace with our mean.
## Nulls will be fixed according to categorical or numerical value.
print(df.describe())

""" Index(['Model Name', 'Engine Type', 'HP or lbs thr ea engine',
       'Max speed Knots', 'Rcmnd cruise Knots', 'Stall Knots dirty',
       'Fuel gal/lbs', 'All eng rate of climb', 'Eng out rate of climb',
       'Takeoff over 50ft', 'Landing over 50ft', 'Empty weight lbs',
       'Length ft/in', 'Wing span ft/in', 'Range N.M.', 'Price'],
      dtype='object') """

# --------------------------------------------------------

"""

To-do List:

1. Fill nulls, correct csv errors 

2. Other Preprocessing:
'HP or lbs thr ea engine'=>float
'Max speed Knots'=>float
'All eng rate of climb'=>float
'Landing over 50ft'=>float
'Empty weight lbs'=>float
'Length ft/in'=>float
'Wing span ft/in'=>float
'Range N.M.'=>float

3. EDA, scatterplots, histograms, line of best fit, correlation scores. Provide insight for feature engineering and feature selection in our model.

"""

# -----------------------------------------------------------------------------------------

print("Number of unique model names:")
print(df['Model Name'].nunique())
# print(df['Max speed Knots'][0:5])
# print(df['Eng out rate of climb'][0:5])
## This is an integer recorded as an object will need to correct to get more accurate model.

## Other objects that could be converted to int/float are: Empty weight lbs, Length ft/in, Wing span ft/in, Range N.M, HP or lbs thr ea engine, etc..
## Once all dtypes are corrected and null values are processed we can get into our data. 
## Model Name has 284 unique values out of 517 samples. This way too many for our categorical data and we will drop the column for now.

## Only real object variables should be Engine Type - categorical variables - will use preprocessing pipeline to separate from numerical variables

## Nulls for numerical columns will be filled with mean, for our categorical variables we will fill with most frequent

# -----------------------------------------------------------------------------------------------------

# LENGTH
df['Length ft/in'] = df['Length ft/in'].astype('string')
df['Length ft/in'] = df['Length ft/in'].fillna('0/0')
df['Length ft/in'] = df['Length ft/in'].replace('N/C', '0/0', regex=True)
df['Length ft/in'] = df['Length ft/in'].replace('/', ',', regex=True)

length = df['Length ft/in']
print(df['Length ft/in'].head())
true_length = []

for i in range(len(length)):
    length_split = []
    length_split = length[i].split(',')
    length_ft = float(length_split[0])
    length_in = float(length_split[1])
    total_length = length_ft + (length_in / 12)
    true_length.append(total_length)

df['Length ft/in'] = true_length
print(df['Length ft/in'])


# WINGSPAN

df['Wing span ft/in'] = df['Wing span ft/in'].astype('string')
df['Wing span ft/in'] = df['Wing span ft/in'].fillna('0/0')
df['Wing span ft/in'] = df['Wing span ft/in'].replace('N/C', '0/0', regex=True)
df['Wing span ft/in'] = df['Wing span ft/in'].replace('/', ',', regex=True)

wing_span = df['Wing span ft/in']
print(wing_span.head())
true_span = []

for i in range(len(wing_span)):
    w_split = []
    w_split = wing_span[i].split(',')
    length_ft = float(w_split[0])
    true_span.append(length_ft)

df['Wing span ft/in'] = true_span
print(df['Wing span ft/in'])

# -----------------------------------------------------------
# MANUALLY CLEANING REST OF OUR COLUMNS

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

## Converting Fuel gal/lbs to same dtype as rest of dataset
df['Fuel gal/lbs'] = df['Fuel gal/lbs'].astype('float')

# -----------------------------------------------------------------
""" Now that data is cleaned, we can do some exploratory data analysis. """
# Price

print(df['Price'].describe())
price_mean = df['Price'].mean()
print(price_mean)
# price mean = $2,362,673.18

price_max = max(df['Price'])
print(price_max)
# price max = $5,100,000.00

price_min = min(df['Price'])
print(price_min)
# min price = $650,000.00

price_std = df['Price'].std()
print(price_std)
# price stdev = $1,018,731.40


## - Scatterplots - 

## Looking to get insights on potential features that may have a stronger correlation to our target, as well as eachother.

# Measurements vs. Price

#plt.scatter(df['Length ft/in'], df['Price'], alpha=0.5, color='blue')
sns.regplot(data=df, x='Length ft/in', y='Price', ci=None, color=".3", line_kws=dict(color="r"))
plt.title("Length vs Price")
plt.xlabel("Length (feet)")
plt.ylabel("Price (Millions of $)")
plt.show()
plt.clf()

#plt.scatter(df['Wing span ft/in'], df['Price'], alpha=0.5, color='orange')
sns.regplot(data=df, x='Wing span ft/in', y='Price', ci=None, color=".3", line_kws=dict(color="r"))
plt.title("Wingspan vs Price")
plt.xlabel("Wingspan (feet)")
plt.ylabel("Price (Millions of $)")
plt.show()
plt.clf()

""" There seems to be a semi-strong correlation between price and length, as well as a weak correlation between price and wingspan. """

# SPEED vs PRICE

#plt.scatter(df['Max speed Knots'], df['Price'], alpha=0.5, color='green')
sns.regplot(data=df, x='Max speed Knots', y='Price', ci=None, color=".3", line_kws=dict(color="r"))
plt.title("Max Speed vs Price")
plt.xlabel("Max Speed (knots)")
plt.ylabel("Price (Millions of $)")
plt.show()
plt.clf()

#plt.scatter(df['Rcmnd cruise Knots'], df['Price'], alpha=0.4, color='purple')
sns.regplot(data=df, x='Rcmnd cruise Knots', y='Price', ci=None, color=".3", line_kws=dict(color="r"))
plt.title("Recommended cruising speed vs Price")
plt.xlabel("Recommended cruising speed (knots)")
plt.ylabel("Price (Millions of $)")
plt.show()
plt.clf()
"""
The scatterplots show a very clear, strong correlation between speed and price.
This includes both Max Speed and Recommended Cruising Speed.

"""

# Fuel vs Price

#plt.scatter(df['Fuel gal/lbs'], df['Price'], alpha=0.5, color='brown')
sns.regplot(data=df, x='Fuel gal/lbs', y='Price', ci=None, color=".3", line_kws=dict(color="r"))
plt.title("Fuel vs Price")
plt.xlabel("Fuel (gal/lbs)")
plt.ylabel("Price (Millions of $)")
plt.show()
plt.clf()

""" 
Some correlation here shown by our regplot that was not present in just the scatterplot.

"""


# ------------------------------------------------------------------------------------------------------
"""
Important features for our target (price) appear to be Fuel gal/lbs, both of our speed columns, and length of plane. 

"""
# ------------------------------------------------------------------------------------------------------

# Checking some of our features correlation to eachother

## Looking at correlation between size and speed

# SPEED vs LENGTH

#plt.scatter(df['Max speed Knots'], df['Length ft/in'], alpha=0.4, color='gray')
sns.regplot(data=df, x='Max speed Knots', y='Length ft/in', ci=None, color=".3", line_kws=dict(color="r"))
plt.title("Max Speed vs Length")
plt.xlabel("Max speed (knots)")
plt.ylabel("Length (feet)")
plt.show()
plt.clf()

#plt.scatter(df['Rcmnd cruise Knots'], df['Length ft/in'], alpha=0.4, color='black')
sns.regplot(data=df, x='Rcmnd cruise Knots', y='Length ft/in', ci=None, color=".3", line_kws=dict(color="r"))
plt.title("Recommended cruising speed vs Length")
plt.xlabel("Recommended cruising speed (knots)")
plt.ylabel("Length (feet)")
plt.show()
plt.clf()

"""
Length does seem to have some average correlation with both max speed and recommended cruising speed.

"""

# SPEED vs WINGSPAN

#plt.scatter(df['Max speed Knots'], df['Wing span ft/in'], alpha=0.4, color='red')
sns.regplot(data=df, x='Max speed Knots', y='Wing span ft/in', ci=None, color=".3", line_kws=dict(color="r"))
plt.title("Max Speed vs Wingspan")
plt.xlabel("Max speed (knots)")
plt.ylabel("Wingspan (feet)")
plt.show()
plt.clf()

#plt.scatter(df['Rcmnd cruise Knots'], df['Wing span ft/in'], alpha=0.4, color='teal')
sns.regplot(data=df, x='Rcmnd cruise Knots', y='Wing span ft/in', ci=None, color=".3", line_kws=dict(color="r"))
plt.title("Recommended cruising speed vs Wingspan")
plt.xlabel("Recommended cruising speed (knots)")
plt.ylabel("Wingspan (feet)")
plt.show()
plt.clf()

""" There appears to be a weak correlation between Wingspan and speed. """

# -----------------------------------------------------------------------------------------------------