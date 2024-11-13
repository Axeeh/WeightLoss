Supervised Learning Weight Loss Project

This project aims to analyze and predict the final weight of participants who followed a diet plan, with
a focus on understanding the influence of various physiological and lifestyle factors. Specifically, two
predictive models are developed: one that includes Basal Metabolic Rate (BMR) and one that
excludes it, allowing us to assess the importance of BMR in predicting final weight.

1. Data Exploration and Preprocessing

In this first step the dataset was examined and prepared for analysis. Data visualization techniques
were employed to uncover underlying patterns and issues with the dataset. No duplicate records
were identified. Missing values were handled using imputation methods like kNN or Linear
Regression. Outliers were detected and removed. After these operations, the dataset is ready for
further analysis.

2. Preparing the Dataset

In Task 2, feature processing steps were implemented to prepare the dataset for modeling:
● Continuous features were standardized to ensure a consistent scale, enhancing

model performance and convergence.
● Categorical features:

○ Binary encoding was applied for binary variables (gender and smoking)
○ Ordinal encoding was used for ordered variables (sleep quality)
○ One-hot encoding was employed for nominal variables (e.g., work sector).

Feature selection methods were subsequently applied to evaluate the possibility of reducing
dimensionality:

● Univariate Feature Selection
● Recursive Feature Elimination
● Model-Based Feature Selection

Following these analysis variables with the lowest ranks were removed from the dataset:
Physical Activity Level, Duration (weeks) and all the Work Sectors

3. Predictive Modeling Approaches
In Task 3, several linear regression models were developed to compare their performance
with and without Basal Metabolic Rate (BMR) as a feature:

a) sklearn Model:

A linear regression model using Sklearn’s
LinearRegression as an initial implementation
for comparison. The model with BMR
consistently outperforms the model without it,
achieving both a higher R² score and a lower
MSE.

b) Batch Gradient Descent:

Linear regression was
performed using a custom
implementation of batch
gradient descent to optimize
model weights.

Various learning rates were
tested to identify the optimal
rate that minimized cost and
achieved the best convergence.
0.1 is the best one

c) Mini-Batch Gradient Descent:

Mini-batch gradient descent was applied to analyze convergence speed and computational efficiency.

Different batch sizes (powers of 2) were tested on different learning rates to evaluate how model
accuracy and convergence varied with batch size. Smaller batch sizes consistently yield the best
performances. Higher sizes result in higher MSE values.

d) Polynomial Feature Augmentation:

Polynomial features were added to the
dataset to capture non-linear
relationships among features.

Batch gradient descent was then
applied to this augmented dataset,
with and without BMR, to assess any
improvements in model accuracy. A
learning rate of 0.01 is optimal for both
models

e) Comparing Learning Curves

Learning curves developed in the previous tasks were compared to identify the optimal approach.
Mini-batch gradient descent appears as the most effective method when working with small batch
sizes and a moderate learning rate.

f) Lasso Regularization with Augmented Dataset:

Lasso regularization was applied to
enforce sparsity and identify the most
predictive features. Various α values
were tested to find an optimal level of
feature selection and model
performance. The models start to
underfit with greater values of alpha.

g) Ridge Regression with Augmented Dataset:

Ridge regression was implemented to
manage feature weights and improve
generalization without enforcing
sparsity. Different regularization
strengths were tested to assess the
impact of BMR on model performance
under Ridge regularization. In these
plots it is shown how the dataset with
BMR is more accurate

4. Stratifying by Gender

The dataset was stratified by gender to enable separate analyses for male and female subsets. This
division allowed for a more targeted evaluation of the predictive factors influencing final weight in
each group. For both subsets, linear regression with batch gradient descent was applied to optimize
model weights.
Although the two plots look similar, this analysis suggests that different features may hold varying
predictive importance across genders. A learning rate of 0.1 was used since in earlier analysis we
proved it is the best one for batch gradient descent models.

5. Best Model?

The results of Task 4 were compared to the global
models from Task 3 to determine whether
stratification by gender provided an advantage in
predictive performance. The analysis showed that
all four models achieved similar R² scores, with
values around 0.8, a similar performance between
gender specific and global models. This suggests
that the inclusion of BMR as a feature captures
key predictive information that benefits both
genders, potentially reducing the need for
separate models. Models without BMR
consistently achieved lower R² scores,
highlighting even more BMR’s importance as a
predictor.

6. Conclusion
The series of analyses highlights the importance of including BMR, using optimal regularization, and
potentially employing gender-specific models for best performance. BMR significantly enhances the
model’s ability to predict the target variable, especially when used with Ridge or Lasso regression at
optimal regularization strengths. Gender-specific analysis reveals that feature relevance can vary
between groups, suggesting the need for demographic stratification in predictive modeling. This
approach allows for capturing subtle differences in how features relate to the target variable,
ultimately leading to more accurate and generalizable models.
