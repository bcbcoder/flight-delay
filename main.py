import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score

data = pd.read_csv('/path/tofile') 

data = data.head(3000)
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


def calculate_prior(data, Y):
    classes = sorted(list(data[Y].unique()))
    prior = []
    for i in classes:
        prior.append(len(data[data[Y] == i]) / len(data))
    return prior


def calculate_likelihood_gaussian(df, feat_name, feat_val, Y, label):
    df = df[df[Y] == label]
    mean, std = df[feat_name].mean(), df[feat_name].std()
    if std == 0:
        std = 1e-6  # Handle division by zero by adding a small epsilon
    p_x_given_y = (1 / (np.sqrt(2 * np.pi) * std)) * np.exp(-((feat_val - mean) ** 2 / (2 * std ** 2)))
    return p_x_given_y


def naive_bayes_gaussian(df, X, Y):
    features = list(df.columns)[:-1]
    prior = calculate_prior(df, Y)

    Y_pred = []

    for x in X:
        labels = sorted(list(df[Y].unique()))
        likelihood = [1] * len(labels)
        for j in range(len(labels)):
            for i in range(len(features)):
                likelihood[j] *= calculate_likelihood_gaussian(df, features[i], x[i], Y, labels[j])

        post_prob = [1] * len(labels)
        for j in range(len(labels)):
            post_prob[j] = likelihood[j] * prior[j]

        Y_pred.append(np.argmax(post_prob))

    return np.array(Y_pred)


train, test = train_test_split(data, test_size=0.1, random_state=41)

X_test = test.iloc[:, :-1].values
Y_test = test.iloc[:, -1].values
Y_pred = naive_bayes_gaussian(train, X=X_test, Y="delay")

print(confusion_matrix(Y_test, Y_pred))
print(f1_score(Y_test, Y_pred))


