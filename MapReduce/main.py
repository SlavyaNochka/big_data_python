import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from Classificator import BayesСlassification

strings = [
    'hello my dear friend i\'m so happy see you',
    'hi you\'re win a special prize and i\'m so want meet you to give it',
    'hello my dear friend you\'re win a special action',
    'hello how a u',
    ['good evening', 'my', 'dear friend']
]

data_link = 'spamdb.csv'

data = pd.read_csv(data_link, encoding='latin-1')
col = data.columns
col = [i for i in col if 'Unnamed' not in i]
data = pd.DataFrame(data, columns=col)

x, y = data.loc[:, col[1]], data.loc[:, col[0]]
y = [1 if i == 'spam' else 0 for i in y]


x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=0)
counts = [y_train.count(0), y_train.count(1)]

bc = BayesСlassification(counts)
bc.stat(x_train, y_train)

print("=" * 5, 'TEST', "=" * 5)

pred = bc.predict_many_msg(x_test)
print(f'accuracy: {accuracy_score(pred, y_test)}')

print("=" * 5, 'WORD SPAM', "=" * 5)

s = ['watch', 'good', 'special', 'prize', 'action', 'is']
for j in s:
    print(f'{j}: {bc.word_spam_probability(j)}')
# redis-cli
#   hgetall s

print("=" * 5, 'MESSAGE SPAM', "=" * 5)

for i in strings:
    print(f'{i}: {bc.predict(i)}')

print("=" * 5, 'PROD', "=" * 5)
pred = bc.predict_many_msg(x)
print(f'accuracy: {accuracy_score(pred, y)}')
