import pandas as pd
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression
dataframe = pd.read_csv("landprice.csv")
inputs = dataframe.drop("price", axis="columns")
outputs = dataframe["price"]
model1 = SVC()
model2 = LinearRegression()
model1.fit(inputs, outputs)
model2.fit(inputs, outputs)
test1 = pd.DataFrame([[250, 4, 0], [300, 3, 1]], columns=inputs.columns)
test2 = pd.DataFrame([[1200, 4, 1], [780, 3, 0]], columns=inputs.columns)
amount = model1.predict(test1)
amountreg = model2.predict(test2)
print(amount)
print(amountreg)
