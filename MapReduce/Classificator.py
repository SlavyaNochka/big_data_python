import re
from collections import defaultdict

class BayesСlassification:
    def __init__(self, counts):
        self.words = defaultdict(lambda: [0, 0, 0,
                                          0])  # [сколько раз слово встречалось в ham, сколько раз слово встречалось в spam, вес в ham, вес в spam]
        self.ph = counts[0] / sum(counts)
        self.ps = 1 - self.ph
        self.counts = counts
        self.learn_in_process = False  # добавить ли в обучение новое предложение

    def stat(self, x, y):
        min_lenght = 2
        x = [re.findall("[a-z'0-9]+", i.lower()) for i in x]
        for i in range(len(y)):
            for word in x[i]:
                if len(word) > min_lenght:
                    self.words[word][y[i]] += 1
        for word in self.words:
            self.words[word][2] = self.words[word][0] / sum([self.words[w][0] for w in self.words])  # вес слова в ham
            self.words[word][3] = self.words[word][1] / sum([self.words[w][1] for w in self.words])  # вес слова в spam

    # Вероятность того, что сообщение, содержащее данное слово W, является спамом
    def word_spam_probability(self, word):
        if sum(self.words[word][2:]) == 0:
            self.words[word][2] = self.ph
            self.words[word][3] = self.ps
        p_ham = self.words[word][2] * self.ph
        p_spam = self.words[word][3] * self.ps
        if p_ham == 0 and p_spam == 0:  # если слово встречено впервые, то его вес в ham/spam будем считать равным ph/ps
            p_ham = self.ph ** 2
            p_spam = self.ps ** 2
        return p_spam / (p_spam + p_ham)

    # Спамовость сообщения - вероятность того, что сообщение, содержащее слова 𝑊1, 𝑊2, . . . ,
    # 𝑊𝑁, является спамом
    def message_spam_probability(self, msg):
        p_ham, p_spam = 1, 1
        for word in msg:
            p_spam *= self.word_spam_probability(word)
            p_ham *= 1 - self.word_spam_probability(word)
            if p_spam == 0 and p_ham == 0:
                p_ham, p_spam = self.ph, self.ps
        p = p_spam / (p_spam + p_ham * ((self.ph / self.ps) ** (1 - len(msg))))
        if self.learn_in_process:
            self.counts[round(p)] += 1
            self.ph = self.counts[0] / sum(self.counts)
            self.ps = 1 - self.ph
            for word in msg:
                if len(word) > 2:
                    self.words[word][round(p)] += 1
                    self.words[word][round(p) + 2] = self.words[word][round(p)] / sum(
                        [self.words[w][round(p)] for w in self.words])
        return p

    def predict(self, msg, learn_in_process=False):
        msg = re.findall("[a-z'0-9]+", ''.join(msg))
        self.learn_in_process = learn_in_process
        return self.message_spam_probability(msg)

    def predict_many_msg(self, x, learn_in_process=False):
        p = []
        for msg in x:
            p.append(round(self.predict(msg, learn_in_process)))
        return p