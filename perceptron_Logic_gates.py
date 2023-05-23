import pandas as pd 
import seaborn as sns
from sklearn.linear_model import Perceptron
import matplotlib.pyplot as plt
import numpy as np
from itertools import product

# step 1 
#############
#inputs to AND Logic
data = [[0,0],[0,1],[1,0],[1,1]]

# step 2
##########
#Labels for AND LOGIC
labels = [0,0,0,1]

#step 3
##############
# plot the points
plt.scatter([point[0] for point in data],[point[1] for point in data],c=labels)
plt.show()

#step 4
##############
#building the Perceptron
#classifier is an object of Perceptron
# max_iter is the no of times the perceptron loops through the training data, Default is 1000, we cut it to 40
classifier = Perceptron(max_iter = 40)

#step 5
############
# Train the model by calling the .fit()method and using data and labels as parameters
classifier.fit(data,labels)

#step 6
############
# print the accuracy of the model
accuracy_model_and= classifier.score(data,labels)

print(accuracy_model_and)

#step 7
# OR gate Logic

data =[[0,0],[0,1],[1,0],[1,1]]
labels =[0,1,1,1]

#step 8
#build the model for OR logic
classifier.fit(data,labels)

# print the accuracy of the model
accuracy_model_or = classifier.score(data,labels)

print(accuracy_model_or)

#step 9
#Visualizing the perceptron
#A decision boundary is the line that determines whether the output should be a 1 or a 0. Points that fall on one side of the line will be a 0 and points on the other side will be a 1.
#The point at [0.5, 0.5] is pretty close to the decision boundary as well.
result = classifier.decision_function([[0, 0], [1, 1], [0.5, 0.5]])
print(result)

#step 10
#creating a list of points we want to input x and y , a list of 100 evenly spaced decimals between 0 and 1
x_values = np.linspace(1,100)
y_values = np.linspace(1,100)

#step 11
#We have a list of 100 x values and 100 y values. product() finds every possible combination of those x and y values.
point_grid = list(product(x_values, y_values))

#step 12
#distances variable stores positive and negative values, considering only about how far away a point is from the boundary and not about the sign.
distances = classifier.decision_function(point_grid)

#step 13
#Taking the absolute value of every distance
abs_distances = [abs(point_grid) for i in distances]


#step 14
#plotting the heat map
distances_matrix = np.reshape(abs_distances, (100,100))

# step 15
heatmap = plt.pcolormesh(x_values,y_values, distances_matrix)
plt.colorbar(heatmap)
plt.show()
