import pandas as pd

df = pd.read_csv('Sort.csv')

FakeNews = df[df['real'] == 0]
OrgNews = df[df['real'] == 1]

# Step 3: Save the filtered DataFrames to new CSV files
FakeNews.to_csv('FakeNews.csv', index=False)
OrgNews.to_csv('OrgNews.csv', index=False)