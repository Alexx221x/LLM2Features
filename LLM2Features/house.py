# Feature: IncomeBracket
# Usefulness: Categorizes the median income into brackets, providing a more interpretable feature for the classification task.
# Input samples: 'MedInc': [3.11, 8.56, 2.50]
df['IncomeBracket'] = pd.cut(df['MedInc'], bins=[0, 2.5, 4.5, 6.5, 8.5, 10], labels=['Low', 'Lower-Mid', 'Middle', 'Upper-Mid', 'High'])
# Feature: AgeGroup
# Usefulness: Converts the numeric house age into categorical age groups, adding a more interpretable feature.
# Input samples: 'HouseAge': [40.0, 5.0, 28.0]
df['AgeGroup'] = pd.cut(df['HouseAge'], bins=[0, 5, 20, 30, 40, 50], labels=['0-5', '5-20', '20-30', '30-40', '40-50'])
# Feature: HighOccupancyFlag
# Usefulness: Creates a flag for high occupancy, which might correlate with certain survival patterns in the data.
# Input samples: 'AveOccup': [2.076596, 4.322322, 3.120987]
df['HighOccupancyFlag'] = (df['AveOccup'] > 3.0)
# Feature: PopDensity
# Usefulness: Calculates the population density, which could be related to survival rates due to varying living conditions.
# Input samples: 'Population': [976.0, 1234.0, 877.0], 'AveRooms': [5.368085, 6.122345, 5.623124]
df['PopDensity'] = df['Population'] / df['AveRooms']
# Feature: BedrmsPerRoom
# Usefulness: Creates a feature representing the average number of bedrooms per room, which might be relevant to survival outcomes.
# Input samples: 'AveBedrms': [1.038298, 0.983741, 1.254321], 'AveRooms': [5.368085, 6.122345, 5.623124]
df['BedrmsPerRoom'] = df['AveBedrms'] / df['AveRooms']
# Explanation why the column Longitude is dropped
# Longitude is being dropped because it does not directly provide useful information for predicting survival, and might add noise.
df.drop(columns=['Longitude'], inplace=True)
# Explanation why the column Latitude is dropped
# Latitude is being dropped because it does not directly provide useful information for predicting survival, and might add noise.
df.drop(columns=['Latitude'], inplace=True)
# Explanation why the column MedHouseVal is dropped
# MedHouseVal is being dropped because it is likely to be highly correlated with MedInc, leading to potential multicollinearity issues.
df.drop(columns=['MedHouseVal'], inplace=True)
# # Explanation why the column AveBedrms is dropped
# # AveBedrms is being dropped because the new feature BedrmsPerRoom encapsulates the necessary information more effectively.
df.drop(columns=['AveBedrms'], inplace=True)
# # Explanation why the column HouseAge is dropped
# # HouseAge is being dropped because the new feature AgeGroup encapsulates the necessary information more effectively.
df.drop(columns=['HouseAge'], inplace=True)
# # Explanation why the column AveOccup is dropped
# # AveOccup is being dropped because the new feature HighOccupancyFlag encapsulates the necessary information more effectively.
df.drop(columns=['AveOccup'], inplace=True)
# Explanation why the column Population is dropped
# Population is being dropped because the new feature PopDensity encapsulates the necessary information more effectively.
df.drop(columns=['Population'], inplace=True)
