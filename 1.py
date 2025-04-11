import pandas as pd
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression

dataframe = pd.read_csv("landprice.csv")
dataframe.replace('.', pd.NA, inplace=True)
dataframe.dropna(inplace=True)
dataframe = dataframe.astype(float)

inputs = dataframe.drop("price", axis="columns")
outputs = dataframe["price"]

model1 = SVR()
model2 = LinearRegression()
model1.fit(inputs, outputs)
model2.fit(inputs, outputs)

area = int(input('Enter area: '))
rooms = int(input('Enter number of rooms: '))
interior = int(input('Enter interior (0 or 1): '))

test1 = pd.DataFrame([[250, 4, 0], [300, 3, 1]], columns=inputs.columns)
test2 = pd.DataFrame([[area, rooms, interior], [780, 3, 0]], columns=inputs.columns)

amount_svr = model1.predict(test1)
amount_lr = model2.predict(test2)

print("SVR Predictions:", amount_svr)
print("Linear Regression Predictions:", amount_lr)
