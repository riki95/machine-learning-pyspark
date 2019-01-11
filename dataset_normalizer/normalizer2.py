import pandas as pd

data = pd.read_csv("bank-full.csv", delimiter=";")

data = pd.get_dummies(data, columns=['job', 'marital', 'education', 'contact', 'day', 'month', 'poutcome'])

data.default = data.default.map({'no': 0, 'yes': 1})
data.housing = data.housing.map({'no': 0, 'yes': 1})
data.loan = data.loan.map({'no': 0, 'yes': 1})

# Normalize between 0 and 1
for c in ['age', 'balance', 'duration', 'pdays', 'campaign', 'previous']:
    data[c] = (data[c] - data[c].min()) / (data[c].max() - data[c].min())

data.to_csv('normalized2.csv', sep=',', index=False)


