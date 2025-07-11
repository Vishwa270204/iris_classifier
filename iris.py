# iris.py

# Step 1: Import Libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Step 2: Load Iris Dataset
iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target'] = iris.target
df['species'] = df['target'].apply(lambda x: iris.target_names[x])

# Streamlit Title
st.title("🌸 Iris Flower Classifier")
st.subheader("Dataset Preview")
st.write(df.head())

# Step 3: Visualize the Data
st.subheader("📊 Pairplot Visualization")
fig = sns.pairplot(df, hue="species")
st.pyplot(fig)

# Step 4: Prepare Data
X = df[iris.feature_names]
y = df['target']

# Step 5: Split Data (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Train the Model using K-Nearest Neighbors
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

# Step 7: Predict on Test Set
y_pred = model.predict(X_test)

# Step 8: Evaluation Results
st.subheader("✅ Model Evaluation")

accuracy = accuracy_score(y_test, y_pred)
st.write(f"**Accuracy:** {accuracy:.2f}")

st.text("Classification Report:")
st.text(classification_report(y_test, y_pred))

st.text("Confusion Matrix:")
st.write(confusion_matrix(y_test, y_pred))


