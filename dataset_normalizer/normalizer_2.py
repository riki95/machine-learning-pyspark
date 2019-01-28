import pandas as pd

data = pd.read_csv("bank-full.csv", delimiter=";")

data = pd.get_dummies(data, columns=['job', 'marital', 'education', 'contact', 'day', 'month', 'poutcome'])

data.default = data.default.map({'no': 0, 'yes': 1})
data.housing = data.housing.map({'no': 0, 'yes': 1})
data.loan = data.loan.map({'no': 0, 'yes': 1})
data.y = data.y.map({'no': 0, 'yes': 1})


# Normalize between 0 and 1
for c in ['age', 'balance', 'duration', 'pdays', 'campaign', 'previous']:
    data[c] = (data[c] - data[c].min()) / (data[c].max() - data[c].min())

cols_at_end = ['y']
data = data[[c for c in data if c not in cols_at_end] + [c for c in cols_at_end if c in data]]

data = data.rename(index=str, columns={"job_admin.": "job_admin"})

# data['weight'] = data['y']
#
# for index, row in data.iterrows():
#     print(index)
#     if (row['y'][index] == 0):
#         row['weight'][index] = 0.883015
#     else:
#         row['weight'][index] = 1.1

# data.to_csv('normalized2.csv', sep=',', index=False)


