# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Modeling libraries
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, RandomizedSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

# Evaluation metrics
from sklearn.metrics import mean_squared_error, r2_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Oversampling for imbalance
from imblearn.over_sampling import SMOTE

# Association rules
from mlxtend.frequent_patterns import apriori, association_rules

# Suppress warnings for cleaner output
import warnings
warnings.filterwarnings('ignore')

# ----------------------------------
# Phase I: Data Loading and Preprocessing
# ----------------------------------

# Load the synthetic dataset
dataset_path = "datasets/synthetic_waste_collection_data.csv"
df = pd.read_csv(dataset_path)

# Data Cleaning
print("Initial Missing Values:\n", df.isnull().sum())

# Separate numeric and non-numeric columns
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

# Fill missing values for numeric columns with their median
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

# Fill missing values for non-numeric columns with their mode
df[non_numeric_cols] = df[non_numeric_cols].fillna(df[non_numeric_cols].mode().iloc[0])

# Drop duplicates
df.drop_duplicates(inplace=True)

# Confirm no missing values remain
print("Remaining Missing Values:\n", df.isnull().sum())

# Convert 'Date' to datetime and extract features
df['Date'] = pd.to_datetime(df['Date'])
df['DayOfWeek'] = df['Date'].dt.dayofweek  # Monday=0, Sunday=6
df['Month'] = df['Date'].dt.month

# Feature Engineering
df['WastePerCapita'] = df['WasteGenerated(kg)'] / (df['PopulationDensity'] + 1e-3)
df['TrafficImpact'] = df['TrafficDensity(vehicles/km)'] * df['TravelTime(minutes)']
df['FuelCost'] = df['DistanceToNextArea(km)'] / df['FuelEfficiency(km/l)']

# Encode categorical variables using LabelEncoder
label_encoders = {}
categorical_cols = ['TrafficCondition', 'WeatherCondition', 'RoadType', 'WasteType', 'PriorityLevel']
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# ----------------------------------
# Phase II: Exploratory Data Analysis (EDA)
# ----------------------------------

# Correlation Heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f", vmin=-1, vmax=1)
plt.title("Correlation Heatmap")
plt.show()

# Distribution of 'TravelTime(minutes)'
plt.figure(figsize=(10, 6))
sns.histplot(df['TravelTime(minutes)'], kde=True)
plt.title("Distribution of Travel Time")
plt.xlabel("Travel Time (minutes)")
plt.show()

# Scatter plot of 'WasteGenerated(kg)' vs 'TravelTime(minutes)'
plt.figure(figsize=(10, 6))
sns.scatterplot(x='WasteGenerated(kg)', y='TravelTime(minutes)', data=df)
plt.title("Waste Generated vs Travel Time")
plt.show()

# ----------------------------------
# Phase III: Regression Analysis
# ----------------------------------

# Prepare X and y for regression
X = df.drop(columns=['Date', 'AreaID', 'TravelTime(minutes)', 'TravelTimeClass'], errors='ignore')
y = df['TravelTime(minutes)']

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Linear Regression
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Regression Metrics
print("\nRegression Metrics:")
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"MSE: {mse:.2f}, R²: {r2:.2f}")

# Plot Predictions vs Actual
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.xlabel("Actual Travel Time")
plt.ylabel("Predicted Travel Time")
plt.title("Actual vs Predicted Travel Time")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.show()

# Residual Analysis
residuals = y_test - y_pred
plt.figure(figsize=(10, 6))
sns.histplot(residuals, kde=True)
plt.title('Residuals Distribution')
plt.xlabel('Residuals')
plt.show()

# ----------------------------------
# Phase IV: Classification Analysis
# ----------------------------------

# Convert 'TravelTime(minutes)' to binary classes (High vs. Low travel time)
median_time = df['TravelTime(minutes)'].median()
df['TravelTimeClass'] = (df['TravelTime(minutes)'] > median_time).astype(int)

# Prepare X and y for classification
X = df.drop(columns=['Date', 'AreaID', 'TravelTime(minutes)', 'TravelTimeClass'], errors='ignore')
y = df['TravelTimeClass']

# Scale features
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Apply SMOTE to balance classes
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

# Function to evaluate classification models
def evaluate_model(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    print(f"\n{model_name} Metrics:")
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
    plt.title(f'Confusion Matrix for {model_name}')
    plt.show()

# Decision Tree
dt = DecisionTreeClassifier(random_state=42)
param_grid_dt = {
    "criterion": ["gini", "entropy"],
    "max_depth": [5, 10, None],
    "min_samples_split": [2, 5, 10]
}
grid_search_dt = GridSearchCV(dt, param_grid_dt, scoring="accuracy", cv=5)
grid_search_dt.fit(X_train_balanced, y_train_balanced)
best_dt = grid_search_dt.best_estimator_
evaluate_model(best_dt, X_test, y_test, "Decision Tree")

# Random Forest
rf = RandomForestClassifier(random_state=42)
param_dist_rf = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
}
random_search_rf = RandomizedSearchCV(rf, param_distributions=param_dist_rf, n_iter=10, cv=5, random_state=42)
random_search_rf.fit(X_train_balanced, y_train_balanced)
best_rf = random_search_rf.best_estimator_
evaluate_model(best_rf, X_test, y_test, "Random Forest")

# SVM
svm = SVC(random_state=42)
param_grid_svm = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf']
}
grid_search_svm = GridSearchCV(svm, param_grid_svm, scoring='accuracy', cv=5)
grid_search_svm.fit(X_train_balanced, y_train_balanced)
best_svm = grid_search_svm.best_estimator_
evaluate_model(best_svm, X_test, y_test, "SVM")

# Cross-Validation Scores
cv_scores_rf = cross_val_score(best_rf, X_scaled, y, cv=5)
print(f"Random Forest Cross-Validated Scores: {cv_scores_rf}")
print(f"Mean CV Accuracy: {cv_scores_rf.mean():.2f}")

# Feature Importance from Random Forest
importances = best_rf.feature_importances_
feature_names = X.columns
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
plt.title('Feature Importances from Random Forest')
plt.show()

# ----------------------------------
# Phase V: Clustering Analysis
# ----------------------------------

# Clustering: K-Means with Optimal Number of Clusters
silhouette_scores = []
K = range(2, 10)
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    score = silhouette_score(X_scaled, kmeans.labels_)
    silhouette_scores.append(score)

# Plot Silhouette Scores
plt.figure(figsize=(10, 6))
plt.plot(K, silhouette_scores, 'bx-')
plt.xlabel('Number of clusters k')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score for Different k')
plt.show()

# Choose optimal k (e.g., k=2 based on highest silhouette score)
optimal_k = K[np.argmax(silhouette_scores)]
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
kmeans.fit(X_scaled)
df['Cluster'] = kmeans.labels_
print(f"Optimal number of clusters: {optimal_k}")
print("K-Means Silhouette Score:", silhouette_score(X_scaled, df['Cluster']))

# Plot Clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='WasteGenerated(kg)', y='PopulationDensity', hue='Cluster', palette='viridis')
plt.title("K-Means Clustering")
plt.show()

# ----------------------------------
# Phase VI: Association Rule Mining
# ----------------------------------

# Convert continuous variables to binary by thresholding at the mean
binary_df = df.copy()
for col in binary_df.columns:
    if binary_df[col].dtype != 'object' and col not in ['Date', 'AreaID', 'Cluster', 'TravelTime(minutes)', 'TravelTimeClass']:
        binary_df[col] = (binary_df[col] > binary_df[col].mean()).astype(int)

# Drop unnecessary columns
binary_df = binary_df.drop(columns=['Date', 'AreaID', 'Cluster', 'TravelTime(minutes)', 'TravelTimeClass'], errors='ignore')

# Convert DataFrame to boolean type
binary_df = binary_df.astype(bool)

# Apply Apriori algorithm
frequent_itemsets = apriori(binary_df, min_support=0.3, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)

# Display Association Rules
print("\nAssociation Rules:")
print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head())

# ----------------------------------
# Additional Visualizations and Analyses
# ----------------------------------

# PCA Analysis
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
df['PCA1'] = X_pca[:, 0]
df['PCA2'] = X_pca[:, 1]

# Plot PCA results with clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(x='PCA1', y='PCA2', hue='Cluster', data=df, palette='Set1')
plt.title('PCA Plot with Clusters')
plt.show()

# Save the processed DataFrame for future use
df.to_csv("datasets/processed_waste_collection_data.csv", index=False)
print("Processed data saved to 'datasets/processed_waste_collection_data.csv'")


