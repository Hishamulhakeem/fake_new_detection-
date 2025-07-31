import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

fakedata = './data/FakeNews.csv'
truedata = './data/TrueNews.csv'

df_fake = pd.read_csv(fakedata)
df_true = pd.read_csv(truedata)

df_val1 = df_fake.tail(10)
df_val2 = df_true.tail(10)

df_val =pd.concat([df_val1,df_val2],axis=1)
df_val.to_csv('Validate.csv')

col_list = ["title","real"]
data = './data/FakeNewsNet.csv'
df = pd.read_csv(data,usecols=col_list)

X=df['title']
y=df['real']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3)

vector = TfidfVectorizer()
xv_train = vector.fit_transform(X_train)
xv_test = vector.transform(X_test)

model = LogisticRegression()

model.fit(xv_train,y_train)
y_pred = model.predict(xv_test)
print(classification_report(y_test,y_pred))