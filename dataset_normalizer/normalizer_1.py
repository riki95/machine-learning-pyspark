import pandas as pd


def string_to_int_dictionary(strings):
    job_addresses = data[strings].unique()
    ip_dict = dict(zip(job_addresses, range(len(job_addresses))))
    print(strings + " mapping: ", ip_dict)
    return ip_dict


data = pd.read_csv("bank-full.csv", delimiter=";")

# Map strings into numbers
data.job = data.job.map(string_to_int_dictionary('job'))
data.marital = data.marital.map(string_to_int_dictionary('marital'))
data.education = data.education.map(string_to_int_dictionary('education'))
data.default = data.default.map(string_to_int_dictionary('default'))
data.housing = data.housing.map(string_to_int_dictionary('housing'))
data.loan = data.loan.map(string_to_int_dictionary('loan'))
data.contact = data.contact.map(string_to_int_dictionary('contact'))
data.month = data.month.map(string_to_int_dictionary('month'))
data.poutcome = data.poutcome.map(string_to_int_dictionary('poutcome'))
data.y = data.y.map(string_to_int_dictionary('y'))

# Normalize between 0 and 1
for c in data.columns:
    data[c] = (data[c] - data[c].min()) / (data[c].max() - data[c].min())

data.to_csv('normalized.csv', sep=',', index=False)
