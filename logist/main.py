import pandas as pd
import numpy as np
import requests
import seaborn as sn
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import itertools
from sklearn.metrics import classification_report, accuracy_score
from Regres import LogisticRegres
import math


# Функция вычисления параметра CFS:
def cfs_selection(data, inf_vars, res_var, num, logs=False):
    corr_matrix = data.corr()
    inf, inf_num = [], []
    for i in itertools.combinations(inf_vars, num):
        numerator, denominator = 0, 0
        for j in i:
            numerator += corr_matrix[res_var].loc[j][0]
        for j in itertools.combinations(i, 2):
            denominator += corr_matrix[j[0]].loc[j[1]]
        inf.append(i)
        inf_num.append(numerator / math.sqrt(num + 2 * denominator))
    return inf[inf_num.index(max(inf_num))]

# Парсим данные:

data_link = 'https://loginom.ru/sites/default/files/blogpost-files/med-diagn.txt'
d = requests.request(url=data_link, method='GET')
d.encoding = d.apparent_encoding
d = d.text
col = d[:d.index('\r')].split('\t')
inf_vars = col[:-1]
res_var = [col[-1]]

d = d[d.index('\n') + 1:].split('\r\n')
data = [i.split('\t') for i in d]
data = pd.DataFrame(data, columns=col).astype(float)

x, y = data.loc[:, inf_vars], data.loc[:, res_var]
x_train, x_val, y_train, y_val = train_test_split(x, y, train_size=0.75, random_state=78)
x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, train_size=0.75, random_state=20)

x_train = np.array(x_train)
y_train = np.array(y_train).transpose()[0]
x_test = np.array(x_test)
y_test = np.array(y_test).transpose()[0]
x_val = np.array(x_val)
y_val = np.array(y_val).transpose()[0]

x_train = preprocessing.normalize(x_train)
x_test = preprocessing.normalize(x_test)
x_val = preprocessing.normalize(x_val)

print("=" * 5 + "FIRST MODEL" + "=" * 5)

lr = LogisticRegres(count_weights=len(inf_vars) + len(res_var))
lr.learn(x_train, y_train, x_test, y_test)

pred = [lr.pred(i) for i in x_val]
# print(lr.graph)
print(f'Result for 1 model: {accuracy_score(pred, y_val)}')

sn.heatmap(pd.concat([pd.DataFrame(data=x_train, columns=inf_vars), pd.DataFrame(data=y_train, columns=res_var)],
                     axis=1).corr(), annot=True)

sn.pairplot(data, hue=res_var[0])

new_inf_vars = [cfs_selection(data, inf_vars, res_var, 6)]
new_inf_vars = [new_inf_vars[0][i] for i in range(len(new_inf_vars[0]))]

x, y = data.loc[:, new_inf_vars], data.loc[:, res_var]

x_train, x_val, y_train, y_val = train_test_split(x, y, train_size=0.75, random_state=77)
x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, train_size=0.75, random_state=77)

x_train = np.array(x_train)
y_train = np.array(y_train).transpose()[0]
x_test = np.array(x_test)
y_test = np.array(y_test).transpose()[0]
x_val = np.array(x_val)
y_val = np.array(y_val).transpose()[0]


x_train = preprocessing.normalize(x_train)
x_test = preprocessing.normalize(x_test)
x_val = preprocessing.normalize(x_val)

new_lr = LogisticRegres(count_weights=len(new_inf_vars)+len(res_var))
new_lr.learn(x_train, y_train, x_test, y_test)

print("=" * 5 + "FIRST MODEL" + "=" * 5)

pred = [new_lr.pred(i) for i in x_val]
print(f'Result for 2 model: {accuracy_score(pred, y_val)}')
