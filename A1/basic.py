import pandas as pd
import matplotlib.pyplot as plt

# (1)	(10 points) Read in the Iris dataset using functions in Pandas package.

data = pd.read_csv('../csv/iris.csv', header=None)
data.columns = ['sepal-length', 'sepal-width', 'petal length', 'petal width', 'class']

# (2)	(20 points) Calculate and print the number of rows and columns that this dataset contains.

print(data.shape)

# (3)	(20 points) Get all the values of the last column and print the distinct values of the last column.

last_column = data.iloc[:, -1].copy()
print("Distinct values of the last column: ")
print(last_column.unique())

# (4)	(25 points) Calculate the number of rows, the average value of the first column, the maximum value of the second
# column, and the minimum value of the third column when the last column has value “Iris-setosa”.

print("when the last column has value Iris-setosa:")
print("Number of rows: " + str(data.loc[data["class"] == "Iris-setosa"].shape[0]))
print("Average of first column:")
print('\t Mean = %.2f' % data.loc[data["class"] == "Iris-setosa"].iloc[:, 0].mean())
print("Maximum value of the second column:")
print('\t Maximum = %.2f' % data.loc[data["class"] == "Iris-setosa"].iloc[:, 1].max())
print("Minimum value of the last column:")
print('\t Minimum = %.2f' % data.loc[data["class"] == "Iris-setosa"].iloc[:, 2].min())

# (5)	(23 points) Draw a scatter plot with the data of the first column and the second column (y axis represents the
# second column and x axis represents the first column). Show the points in different colors and shapes when the last
# column’s values are different.
# Draw scatter plot for 3 classes in the same figure
plt.clf()

plt.scatter(data.loc[data["class"] == "Iris-setosa", "sepal-length"],
            data.loc[data["class"] == "Iris-setosa", "sepal-width"], color='red')
plt.scatter(data.loc[data["class"] == "Iris-versicolor", "sepal-length"],
            data.loc[data["class"] == "Iris-versicolor", "sepal-width"], color='blue')
plt.scatter(data.loc[data["class"] == "Iris-virginica", "sepal-length"],
            data.loc[data["class"] == "Iris-virginica", "sepal-width"], color='yellow')
plt.xlabel("sepal-length")
plt.ylabel("sepal-width")
plt.legend(['Iris-setosa', 'Iris-versicolor', "Iris-virginica"])
plt.title("Iris data set")

plt.show()

# print scatter plot in 3 different figure
plt.clf()

fig, (axe1, axe2, axe3) = plt.subplots(1, 3, figsize=(12, 4))

axe1.scatter(data.loc[data["class"] == "Iris-setosa", "sepal-length"],
             data.loc[data["class"] == "Iris-setosa", "sepal-width"], color='red')
axe1.legend(['Iris-setosa'])
axe1.set_xlabel("sepal-length")
axe1.set_ylabel("sepal-width")

axe2.scatter(data.loc[data["class"] == "Iris-versicolor", "sepal-length"],
             data.loc[data["class"] == "Iris-versicolor", "sepal-width"], color='blue')
axe2.legend(['Iris-versicolor'])
axe2.set_xlabel("sepal-length")
axe2.set_ylabel("sepal-width")

axe3.scatter(data.loc[data["class"] == "Iris-virginica", "sepal-length"],
             data.loc[data["class"] == "Iris-virginica", "sepal-width"], color='yellow')
axe3.legend(['Iris-virginica'])
axe3.set_xlabel("sepal-length")
axe3.set_ylabel("sepal-width")

plt.show()
