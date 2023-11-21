import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, f1_score

data = pd.read_csv('/Users/bernardc.burman/Library/Containers/com.microsoft.Excel/Data/Downloads/AirTrax_Flight.csv') # get rid of file path before git upload

data = data.head(1000)
data = data.dropna()
data = data.drop(columns=['Carrier Code', 'Flight Number', 'Actual departure time', 'Taxi-Out time (Minutes)',
                          'Wheels-off time', 'Delay Carrier (Minutes)', 'Delay Weather (Minutes)',
                          'Delay National Aviation System (Minutes)', 'Delay Security (Minutes)',
                          'Delay Late Aircraft Arrival (Minutes)'])
data['month'] = data['Date (MM/DD/YYYY)'].str.split('/').str[0]
data['month'] = data['month'].astype(int)
data = data.drop(columns=['Date (MM/DD/YYYY)'])
data['Tail Number'] = data['Tail Number'].str.replace('N', '').str.replace('FR', '')
data['Tail Number'] = pd.to_numeric(data['Tail Number'], errors='coerce')
data = data.dropna(subset=['Tail Number'])
data['isDelayed'] = data['Departure delay (Minutes)'] + data['Actual elapsed time (Minutes)'] - data[
    'Scheduled elapsed time (Minutes)']
data = data.drop(columns=['Actual elapsed time (Minutes)'])
data['Scheduled departure hour'] = data['Scheduled departure time'].str.split(':').str[0].astype(int)
data = data.drop(columns=['Scheduled departure time'])

airport_mapping = {}
unique_id = [0]


def map_airport_code(airport_code):
    if airport_code not in airport_mapping:
        airport_mapping[airport_code] = unique_id[0]
        unique_id[0] += 1
    return airport_mapping[airport_code]


data['Destination Airport ID'] = data['Destination Airport'].apply(map_airport_code)
data = data.drop(columns=['Destination Airport'])


category_mapping = {
    1: 1,
    2: 1,
    3: 1,
    4: 1,
    5: 1,
    6: 1,
    7: 2,
    8: 2,
    9: 2,
    10: 2,
    11: 2,
    12: 2,
    13: 3,
    14: 3,
    15: 3,
    16: 3,
    17: 3,
    18: 3,
    19: 4,
    20: 4,
    21: 4,
    22: 4,
    23: 4,
    24: 4
}


data['Scheduled departure hour category'] = data['Scheduled departure hour'].map(category_mapping)
data = data.drop(columns=['Scheduled departure hour'])

category_mapping2 = {
    1: 1,
    2: 1,
    3: 1,
    4: 2,
    5: 2,
    6: 2,
    7: 3,
    8: 3,
    9: 3,
    10: 4,
    11: 4,
    12: 4
}


data['Month Collapsed'] = data['month'].map(category_mapping2)
data = data.drop(columns=['month'])


def determine_delay(isDel):
    if isDel > 25:
        return 1  # Flight is delayed
    else:
        return 0


data['delay'] = data['isDelayed'].apply(determine_delay)
data = data.drop(columns=['isDelayed'])

print(data.iloc[10])
print(data.head(30))

print(data.head(10))


correlation_matrix = data.corr(method='pearson')


plt.figure(figsize=(10, 8))


sns.set(font_scale=1)
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=.5)

plt.title('Pearson Correlation Heatmap')  # Set the heatmap's title (optional)

plt.show()

train, test = train_test_split(data, test_size=0.2, random_state=41)

# Define the features and target variable
X_train = train.iloc[:, :-1].values
Y_train = train.iloc[:, -1].values
X_test = test.iloc[:, :-1].values
Y_test = test.iloc[:, -1].values

# Create a Random Forest classifier
rf_classifier = RandomForestClassifier(random_state=41)

param_grid = {
    'n_estimators': [50, 100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(rf_classifier, param_grid, cv=5, scoring='f1')

grid_search.fit(X_train, Y_train)

best_params = grid_search.best_params_

best_rf_classifier = RandomForestClassifier(**best_params, random_state=41)

best_rf_classifier.fit(X_train, Y_train)

Y_pred = best_rf_classifier.predict(X_test)

print(confusion_matrix(Y_test, Y_pred))
print(f1_score(Y_test, Y_pred))


