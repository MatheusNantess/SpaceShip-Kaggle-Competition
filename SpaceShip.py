import pandas as pd 
import os
import matplotlib.pyplot as plt
import seaborn as sns
new_directory = 'C:/Users/mathe/OneDrive/Pictures/Python/CsvExcelFiles'
os.chdir(new_directory)

df = pd.read_csv('train.csv')
df.head()




# --------- Exploratory Data Analysis ------------ #  
#1 -  What are the categorical variables?
catergorical_variables = df.select_dtypes(include='object').columns
for cat_col in catergorical_variables:
    print(cat_col)
print(df.info())
#2 -  What are the numerical variables?

numerical_variables = df.select_dtypes(include=['int64','float64']).columns
for num_col in numerical_variables:
    print(num_col)

#3 - Check for unique values in the categorical variables 
for cat_col in catergorical_variables:
    print(cat_col, df[cat_col].nunique())
#4 - Verify the describe() method
print(df.describe())
df.head()
#4.1 obs: there is age equals to 0, which is not possible
#6 - verify the occurrence of missing values
print(df.isnull().sum())
#6.1 - there are missing values in the columns Age, HomePlanet, CryoSleep and VIP
sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='viridis')


#7 - evaluate variables individually through bar plots, histograms and countplots
#7.1 - Age is a normal distribution
sns.histplot(df['Age'],kde=True)
#7.2 - destination is not a normal distribution
sns.countplot(x='Destination',data=df)
#7.3 - HomePlanet is not a normal distribution
sns.countplot(x='HomePlanet',data=df)
#7.5 - CryoSleep is not a normal distribution
sns.countplot(x='CryoSleep',data=df)
#7.4 - VIP is not a normal distribution
sns.countplot(x='VIP',data=df)

#spending columns
sns.boxplot(df[['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']])
plt.yscale('log')
#8 - view the balance of the target variable
#8.1 - its a balanced target variable
sns.countplot(x='Transported',data=df)

#9 - Analyze every variable relationship with the target variable 
#9.1 - Age and Transported
sns.histplot(x='Age',hue='Transported',data=df)

#9.2 - Destination and Transported
sns.countplot(x='Destination',hue='Transported',data=df)
#9.3 - HomePlanet and Transported
sns.countplot(x='HomePlanet',hue='Transported',data=df)
#9.4 - CryoSleep and Transported
sns.countplot(x='CryoSleep',hue='Transported',data=df)
#9.5 - VIP and Transported  
sns.countplot(x='VIP',hue='Transported',data=df)

#10 -  Check for possible outliers using boxplots
for num_col in numerical_variables:
    sns.boxplot(df[num_col])
    plt.show()
    plt.yscale('log')


# ------------ Data Cleaning --------------- #
#1 - Clean the dataset with only the variables that will be used in the model
#1.1 eliminate PassengerId, Name and Cabin
df_clean = df.drop(['PassengerId','Name','Cabin'],axis=1)
df_clean.head()

#2 - Treat the missing values
#2.1 - fill the missing values with the mean
print(df_clean.isnull().sum())

#2.2 - usedropna to eliminate the missing values in the columns VIP, Destination, HomePlanet and CryoSleep
df_clean = df_clean.dropna(subset=['VIP','Destination','HomePlanet','CryoSleep'])

#2.3 - checkinh the corr between Age column based in the column HomePlanet (check out the means)
df.groupby('HomePlanet')['Age'].mean()
sns.boxplot(df_clean, y='Age',x='HomePlanet') 

#2.4  function to fill the missing values in the Age column

def impute_age(cols):
    Age = cols[0]
    HomePlanet = cols[1]

    if pd.isnull(Age):
        if HomePlanet == 'Earth':
            return 24
        elif HomePlanet == 'Mars':
            return 29
        elif HomePlanet == 'Europa':
            return 34
    elif Age == 0:
        if HomePlanet == 'Earth':
            return 24
        elif HomePlanet == 'Mars':
            return 29
        elif HomePlanet == 'Europa':
            return 34
    return Age

#2.5 apply the function to the Age column

#2.6 there is more 0 in the Age column
#2.7 EXTRA: first I will make the predict without removing the 0 values from Age columns and then re run the model removing the 0 values
df_clean['Age'] = df_clean[['Age','HomePlanet']].apply(impute_age,axis=1)
df_clean[df_clean['Age'].isnull()]
df_clean[df_clean['Age'] == 0]

#3 - Treat the outliers
df_clean.describe()

#4 - Treat the categorical variables
#4.1 - use label encoder to transform the categorical variables into numerical variables
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
df_clean['Destination'] = labelencoder.fit_transform(df_clean['Destination'])
df_clean['HomePlanet'] = labelencoder.fit_transform(df_clean['HomePlanet'])
df_clean['CryoSleep'] = labelencoder.fit_transform(df_clean['CryoSleep'])
df_clean['VIP'] = labelencoder.fit_transform(df_clean['VIP'])
df_clean.head()
df_clean.fillna(0,inplace=True)

#5 -  Normalize or padronize the numerical variables

#--------------- Time To Model ----------------#
#1 - separet the target variable from the features
#1.1 first with padronized data
X = df_clean.drop('Transported',axis=1)
y= df_clean['Transported']


#2 - split the data into train and test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=101)

#3 - normalizar as variaveis numericas
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


#4 - train the model using Logistic Regression
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train_scaled, y_train)
#5 - make predictions on the test data
y_pred = model.predict(X_test_scaled)
#6 - evaluate the model performance
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

