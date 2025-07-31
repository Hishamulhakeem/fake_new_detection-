"""
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

data = './data/FakeNewsNet.csv'
df=pd.read_csv(data)

X = df.iloc[:,:5].values
y = df.iloc[:, :].values

print(X.shape)
print(y.shape)

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.5)

model = LogisticRegression()
y_pred = model.fit(X_train,y_train)

outcome1 = model.predict(y_test,y_pred)

print(outcome1)

print(accuracy_score(y_test,y_pred))
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

fakeData = './data/FakeNews.csv'
trueData = './data/TrueNews.csv'

df_fake = pd.read_csv(fakeData)
df_true = pd.read_csv(trueData)



df_fake_val = df_fake.tail(10)
for i in range(5754,5744,-1):
    df_fake.drop([i], axis = 0, inplace = True)
    
df_true_val = df_true.tail(10)
for i in range(17440,17430,-1):
    df_true.drop([i], axis = 0, inplace = True)

df_manual_testing = pd.concat([df_fake_val,df_true_val], axis = 0)
df_manual_testing.to_csv("Test.csv")

df_merge = pd.concat([df_fake, df_true], axis =0 )

df=df_merge.drop(["news_url","tweet_num","source_domain"],axis=1)

x=df["title"]
y=df["real"]

X_train,X_test,y_train,y_test = train_test_split(x,y,test_size=0.2)

vectorization = TfidfVectorizer()
xv_train = vectorization.fit_transform(X_train)
xv_test = vectorization.transform(X_test)

model1 = LogisticRegression()
model1.fit(xv_train,y_train)

y_pred = model1.predict(xv_test)

from sklearn.tree import DecisionTreeClassifier

model2 = DecisionTreeClassifier()
model2.fit(xv_train,y_train)

y_pred = model2.predict(xv_test)

def output_lable(n):
    if n == 0:
        return "Fake News"
    elif n == 1:
        return "Not A Fake News"
    
def manual_testing(news):
    testing_news = {"text":[news]}
    new_def_test = pd.DataFrame(testing_news)
    new_x_test = new_def_test["text"]
    new_xv_test = vectorization.transform(new_x_test)
    pred_LR = model1.predict(new_xv_test)
    pred_DT = model2.predict(new_xv_test)

    return print("\n\nLR Prediction: {} \nDT Prediction: {} ".format(output_lable(pred_LR[0]),output_lable(pred_DT[0])))

news = str(input())
manual_testing(news)





