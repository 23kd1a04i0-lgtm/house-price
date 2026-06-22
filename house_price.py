import pandas as pd
from sklearn.linear_model import LinearRegression

data = pd.read_csv("house_data.csv")

X = data[['Area']]
y = data['Price']

model = LinearRegression()
model.fit(X, y)

area = int(input("Enter House Area: "))
price = model.predict([[area]])

print("Predicted House Price =", price[0])
