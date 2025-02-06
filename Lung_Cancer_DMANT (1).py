#!/usr/bin/env python
# coding: utf-8

# In[55]:


# General Libraries
import pandas as pd
import numpy as np

# Data Visualization
import seaborn as sns
import matplotlib.pyplot as plt

# Machine Learning
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier

# SMOTE for Imbalance Handling
from imblearn.over_sampling import SMOTE


# In[56]:


# Load the dataset
file_path = '/Users/prathibhadevkar/Desktop/Data Science - Course/DMANTI_PROJ/latest_dataset.csv'
data = pd.read_csv(file_path)

# Display basic information and the first few rows of the dataset
data.info(), data.head()


# In[57]:


# Remove unnecessary columns
data_cleaned = data.drop(columns=['index', 'Patient Id'])

# Verify if there are missing values or inconsistent data
missing_values = data_cleaned.isnull().sum()

# Check unique values for the target variable
target_values = data_cleaned['Level'].unique()

missing_values, target_values


# In[58]:


import matplotlib.pyplot as plt
import seaborn as sns

# Plotting the distribution of each feature against the target variable
features = data_cleaned.columns[:-1]  # Exclude the target variable 'Level'

# Create a grid of plots
fig, axes = plt.subplots(nrows=6, ncols=4, figsize=(20, 20))
axes = axes.flatten()

for i, feature in enumerate(features):
    sns.boxplot(data=data_cleaned, x='Level', y=feature, ax=axes[i], palette="viridis")
    axes[i].set_title(f"{feature} vs Level")
    axes[i].set_xlabel("")
    axes[i].set_ylabel(feature)

# Adjust layout
plt.tight_layout()
plt.show()


# In[ ]:





# In[59]:


# Encode the target variable for correlation analysis
data_encoded = data_cleaned.copy()
data_encoded['Level'] = data_encoded['Level'].map({'Low': 0, 'Medium': 1, 'High': 2})

# Compute correlation matrix
correlation_matrix = data_encoded.corr()

# Plot heatmap for correlations
plt.figure(figsize=(16, 12))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title("Feature Correlation Heatmap")
plt.show()



# In[60]:


from itertools import combinations

# Analyze correlations to identify potential feature pairs for interactions
correlation_with_target = correlation_matrix['Level'].sort_values(ascending=False)

# Select features with significant correlation to the target variable
# (Setting a threshold to focus on features with higher influence)
significant_features = correlation_with_target[correlation_with_target.abs() > 0.3].index.tolist()
significant_features.remove('Level')  # Exclude the target variable itself

# Generate combinations of these features for potential interaction terms
interaction_candidates = list(combinations(significant_features, 2))

# Display the selected features and candidate interactions
significant_features, interaction_candidates[:5]  # Show first 5 candidates as a preview


# In[61]:


# Create a list to store interaction terms
interaction_data = []

# Generate interaction terms
for feat1, feat2 in interaction_candidates:
    interaction_data.append(data_cleaned[feat1] * data_cleaned[feat2])

# Convert the list of interaction terms into a DataFrame
interaction_df = pd.DataFrame(interaction_data).T  # Transpose to match original DataFrame structure

# Rename columns appropriately
interaction_df.columns = [f"{feat1}_x_{feat2}" for feat1, feat2 in interaction_candidates]

# Concatenate the interaction DataFrame with the original DataFrame
data = pd.concat([data_cleaned, interaction_df], axis=1)

# To de-fragment the DataFrame
data = data.copy()

# Print the updated DataFrame
# print(data)


# In[13]:


# Split dataset into features and target

X = data.drop(columns=['Level'])
y = data['Level']
X, y


# In[62]:


for column in X.columns:
    crosstab = pd.crosstab(X[column], y, normalize='index')
    crosstab.plot(kind='bar', stacked=True, figsize=(8, 6))
    plt.title(f'{column} vs. Level')
    plt.xlabel(column)
    plt.ylabel('Proportion')
    plt.legend(title='Level')
    plt.show()


# In[16]:


# Use SMOTE to oversample the minority class


smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)
X_resampled, y_resampled


# In[17]:


X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)


# In[63]:


# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define models and parameters
models = {
    'SVC': SVC(probability=True),
    'Random Forest': RandomForestClassifier(),
    'Gradient Boosting': GradientBoostingClassifier(),
    'Decision Tree' : DecisionTreeClassifier()
}

param_grids = {
    'SVC': {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']},
    'Random Forest': {'n_estimators': [100, 200], 'max_depth': [10, 20]},
    'Gradient Boosting': {'learning_rate': [0.01, 0.1], 'n_estimators': [100, 200]},
    'Decision Tree': {'max_depth': [5, 10, 20], 'min_samples_split': [2, 5, 10], 'criterion': ['gini', 'entropy']}
}

# Hyperparameter tuning and evaluation
best_models = {}
for model_name, model in models.items():
    print(f"Training {model_name}...")
    grid_search = GridSearchCV(model, param_grids[model_name], cv=5, scoring='accuracy')
    grid_search.fit(X_train_scaled, y_train)
    best_models[model_name] = grid_search.best_estimator_
    print(f"Best Parameters for {model_name}: {grid_search.best_params_}")
    print(f"Cross-validated Accuracy: {grid_search.best_score_} \n")
    



# In[33]:


# Evaluate models on the test set
for model_name, model in best_models.items():
    print(f"\n{model_name} Evaluation:")
    y_pred = model.predict(X_test_scaled)
    print(classification_report(y_test, y_pred))
#     print("Confusion Matrix:")
#     print(confusion_matrix(y_test, y_pred))


# In[64]:


from sklearn.ensemble import VotingClassifier

# Combine the best-performing models into a voting classifier
voting_clf = VotingClassifier(
    estimators=[
        ('rf', best_models['Random Forest']),
        ('gb', best_models['Gradient Boosting'])
    ],
    voting='soft'  # Use probabilities for weighted voting
)

# Train and evaluate the voting classifier
voting_clf.fit(X_train_scaled, y_train)
y_pred_voting = voting_clf.predict(X_test_scaled)

# Classification Report for Voting Classifier
print("Voting Classifier Evaluation:")
print(classification_report(y_test, y_pred_voting))
print("Confusion Matrix:")
ConfusionMatrixDisplay.from_estimator(voting_clf, X_test_scaled, y_test, cmap="Blues")
plt.title("Confusion Matrix: Voting Classifier")
plt.show()


# In[65]:


# Plot confusion matrix
from sklearn.metrics import ConfusionMatrixDisplay

for model_name, model in best_models.items():
    print(f"\nVisualizations for {model_name}")
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]

    # Confusion Matrix
    ConfusionMatrixDisplay.from_estimator(model, X_test_scaled, y_test, cmap="Blues")
    plt.title(f"Confusion Matrix: {model_name}")
    plt.show()



# In[ ]:


model = best_models['Gradient Boosting']
importance = pd.Series(model.feature_importances_, index=X.columns)
print(importance.sort_values(ascending=False))


# In[32]:


for model_name in ['Random Forest', 'Gradient Boosting']:
    model = best_models[model_name]
    if hasattr(model, 'feature_importances_'):
        # Sort and normalize feature importance
        feature_importance = pd.Series(model.feature_importances_, index=X.columns)
        feature_importance = feature_importance.sort_values(ascending=False)
        
        # Normalize to percentages
        feature_importance_percentage = (feature_importance / feature_importance.sum()) * 100

        # Plot only the top N features
        top_n = 15 # Display top 10 features
        plt.figure(figsize=(10, 6))
        sns.barplot(
            x=feature_importance_percentage.values[:top_n],
            y=feature_importance_percentage.index[:top_n],
            palette="coolwarm"
        )
        
        # Add data labels
        for i, val in enumerate(feature_importance_percentage.values[:top_n]):
            plt.text(val, i, f"{val:.2f}%", fontsize=10, va='center')

        # Add titles and labels
        plt.title(f"Top {top_n} Feature Importances: {model_name}", fontsize=16)
        plt.xlabel("Importance (%)", fontsize=14)
        plt.ylabel("Feature", fontsize=14)
        plt.tick_params(labelsize=12)
        plt.tight_layout()
        plt.show()



# In[52]:


import matplotlib.pyplot as plt

# Model names and their corresponding accuracies
models = ['SVC', 'Random Forest', 'Gradient Boosting', 'Decision Tree', 'Neural Network']
accuracies = [0.9541666666666666, 0.9641666666666668, 0.9505952380952382, 0.9108333333333334, 0.9405]

# Plotting the accuracies with vertical bars
plt.figure(figsize=(10, 6))
plt.bar(models, accuracies, color=['skyblue', 'olive', 'yellow', 'grey', 'violet'])

# Adding details to the plot
plt.title('Model Accuracy Comparison', fontsize=16)
plt.xlabel('Models', fontsize=14)
plt.ylabel('Accuracy', fontsize=14)
plt.ylim(0.9, 1.0)  # Setting y-axis range for better visualization
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# Displaying the plot
plt.tight_layout()
plt.show()



# In[ ]:





# In[ ]:




