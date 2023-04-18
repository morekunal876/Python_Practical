##1.a.	Design a simple machine learning model to train the training instances and test the same.

# python library to generate random numbers
from random import randint

# the limit within which random numbers are generated
TRAIN_SET_LIMIT = 1000

# to create exactly 100 data items
TRAIN_SET_COUNT = 100

# list that contains input and corresponding output
TRAIN_INPUT = list()
TRAIN_OUTPUT = list()

# loop to create 100 data items with three columns each
for i in range(TRAIN_SET_COUNT):
	a = randint(0, TRAIN_SET_LIMIT)
	b = randint(0, TRAIN_SET_LIMIT)
	c = randint(0, TRAIN_SET_LIMIT)

# creating the output for each data item
	op = a + (2 * b) + (3 * c)
	TRAIN_INPUT.append([a, b, c])

# adding each output to output list
	TRAIN_OUTPUT.append(op)

# Sk-Learn contains the linear regression model
from sklearn.linear_model import LinearRegression

# Initialize the linear regression model
predictor = LinearRegression(n_jobs =-1)

# Fill the Model with the Data
predictor.fit(X = TRAIN_INPUT, y = TRAIN_OUTPUT)

# Random Test data
X_TEST = [[ 10, 20, 30 ]]

# Predict the result of X_TEST which holds testing data
outcome = predictor.predict(X = X_TEST)

# Predict the coefficients
coefficients = predictor.coef_

# Print the result obtained for the test data
print('Outcome : {}\nCoefficients : {}'.format(outcome, coefficients))


