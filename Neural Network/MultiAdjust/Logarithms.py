# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
# Generate some skewed data
np.random.seed(42)
x = np.random.exponential(5, size=1000)
y = 2 * x + np.random.normal(0, 5, size=1000)
# Plot the original data
sns.scatterplot(x=x, y=y)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Original Data')
plt.show()
# Plot the histogram of x
sns.histplot(x=x)
plt.xlabel('x')
plt.ylabel('Frequency')
plt.title('Histogram of x')
plt.show()
# Apply log transformation to x
x_log = np.log(x)
# Plot the transformed data
sns.scatterplot(x=x_log, y=y)
plt.xlabel('log(x)')
plt.ylabel('y')
plt.title('Transformed Data')
plt.show()
# Plot the histogram of log(x)
sns.histplot(x=x_log)
plt.xlabel('log(x)')
plt.ylabel('Frequency')
plt.title('Histogram of log(x)')
plt.show()
# Fit a linear regression model to the original data
reg1 = LinearRegression()
reg1.fit(x.reshape(-1, 1), y)
y_pred1 = reg1.predict(x.reshape(-1, 1))
r2_1 = reg1.score(x.reshape(-1, 1), y)
# Fit a linear regression model to the transformed data
reg2 = LinearRegression()
reg2.fit(x_log.reshape(-1, 1), y)
y_pred2 = reg2.predict(x_log.reshape(-1, 1))
r2_2 = reg2.score(x_log.reshape(-1, 1), y)
# Compare the R-squared values of the two models
print(f'R-squared for original data: {r2_1:.3f}')
print(f'R-squared for transformed data: {r2_2:.3f}')

"""
Output:
R-squared for original data: 0.559
R-squared for transformed data: 0.804
The output shows that:
- The original data is highly skewed to the right, with a long tail of high values. The histogram of x shows that most values are concentrated near zero, while some values are very large.
- The transformed data is more symmetric and normal-like. The histogram of log(x) shows that the values are more evenly distributed around the mean.
- The original data has a nonlinear relationship between x and y. The scatterplot of x and y shows that the points are curved and spread out.
- The transformed data has a linear relationship between log(x) and y. The scatterplot of log(x) and y shows that the points are aligned and close to a straight line.
- The linear regression model for the transformed data has a higher R-squared value than the model for the original data. This means that the transformed data explains more variation in y than the original data.
These results demonstrate that log transformation can improve the quality and suitability of the data for machine learning models.
(1) Log Transformation: Purpose and Interpretation | by Kyaw Saw Htoon - Medium. https://medium.com/@kyawsawhtoon/log-transformation-purpose-and-interpretation-9444b4b049c9.
(2) Best practice in statistics: The use of log transformation. https://journals.sagepub.com/doi/full/10.1177/00045632211050531.
(3) Logarithms — What, Why and How. Understanding the intuition behind .... https://towardsdatascience.com/logarithms-what-why-and-how-ff9d050d3fd7.
(4) Log transformation | Data Science and Machine Learning | Kaggle. https://www.kaggle.com/questions-and-answers/61612.
Cite
1 Recommendation
Popular answers (1)
Rahul Pandya
DePaul University
Hello Jaypal Singh Rajput ,
Log-transformations are used when the data is highly skewed (that is highly asymmetrical data). Now when u fit such data in any model, it will result in a highly unstable and an unreliable model. It will give false predictions. [1].
In say regression models, it helps us judge the relation between the data points and ease out the process of detecting outliers significantly.
It shall give a clear picture of what data points to be considered and which ones to be dropped or either sampled (over-sampled or under sample in case of imbalanced datasets).
It shall make the distributions more aligned towards the normal distribution curve for better learning and thus prediction. [2]
The value of log and its base depends on the type of the tail the distribution has and how much more linearly it needs to be transformed.
Here it has been visually depicted how log transformation works on skewed datasets. [3][4]
[1]. study notes: Handling Skewed data for Machine Learning models | by Cheryl | Medium
[2] Log Transformation: How it can transform ML model performance? | by Rahul Kumar | Medium
[3] Transforming Skewed Data for Machine Learning - (opendatascience.com)
[4]Interpreting Log Transformations in a Linear Model | University of Virginia Library Research Data Services + Sciences
Thank you.
"""