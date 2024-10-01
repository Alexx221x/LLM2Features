# (Title and its significance)
# Usefulness: Title often indicates social status and family role, which can influence survival rates (e.g., "Mrs." might indicate women and children who had higher survival rates).
# Input samples: (Three samples of the columns used in the following code, e.g. 'Name': ['Faunthorpe, Mrs. Lizzie (Elizabeth Anne Wilkin', 'Heikkinen, Miss. Laina', 'Moran, Mr. James'])
df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
df['Title'] = df['Title'].replace(['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
df['Title'] = df['Title'].replace('Mlle', 'Miss')
df['Title'] = df['Title'].replace('Ms', 'Miss')
df['Title'] = df['Title'].replace('Mme', 'Mrs')
df['Title'] = df['Title'].astype('category').cat.codes
# (Family Size and its significance)
# Usefulness: Family size can impact survival as families might stay together and help each other, or have difficulties staying together in a crisis.
# Input samples: (Three samples of the columns used in the following code, e.g. 'SibSp': [1, 0, 1], 'Parch': [0, 0, 2])
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
# (IsAlone flag)
# Usefulness: Being alone or not could affect survival rates; those alone might be less likely to survive due to lack of support.
# Input samples: (Three samples of the columns used in the following code, e.g. 'FamilySize': [2, 1, 4])
df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
# (Age categories)
# Usefulness: Age is crucial for survival rates as children were prioritized for lifeboats.
# Input samples: (Three samples of the columns used in the following code, e.g. 'Age': [29.0, 22.0, 35.0])
df['AgeCat'] = pd.cut(df['Age'], bins=[0, 12, 18, 35, 60, 80], labels=[0, 1, 2, 3, 4])
# (Fare per person)
# Usefulness: The cost of the fare per person could indicate the wealth and potentially better access to lifeboats.
# Input samples: (Three samples of the columns used in the following code, e.g. 'Fare': [26.0, 7.25, 71.2833], 'FamilySize': [2, 1, 4])
df['FarePerPerson'] = df['Fare'] / df['FamilySize']
# Explanation why the column 'Cabin' is dropped
# The 'Cabin' column has too many missing values and the cabin number itself might not be significant after extracting the deck.
df.drop(columns=['Cabin'], inplace=True)
# Explanation why the column 'Ticket' is dropped
# The 'Ticket' column contains too many unique values and does not contribute to distinguishing survival rates in a meaningful way.
df.drop(columns=['Ticket'], inplace=True)
# Explanation why the column 'Name' is dropped
# The 'Name' column is no longer needed as we've extracted useful information (Title) from it.
df.drop(columns=['Name'], inplace=True)
# Explanation why the column 'Age' is dropped
# The 'Age' column has been binned into categorical intervals that are more useful for classification.
df.drop(columns=['Age'], inplace=True)
# Explanation why the column 'Fare' is dropped
# The 'Fare' column has been normalized by family size into 'FarePerPerson'.
df.drop(columns=['Fare'], inplace=True)
# Explanation why the column 'Embarked' is dropped
# The 'Embarked' column has a negligible impact on survival prediction in this context and is not critical for our model.
df.drop(columns=['Embarked'], inplace=True)
