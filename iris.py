# iris_normal.py

# Step 1: Import Libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Step 2: Load Iris Dataset
iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target'] = iris.target
df['species'] = df['target'].apply(lambda x: iris.target_names[x])

# Step 3: Visualize the Data
print("\n📊 Generating Pairplot (will open in a separate window)...")
sns.pairplot(df, hue="species")
plt.show()

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
print("\n✅ Model Evaluation")
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}\n")
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Step 9: User Input for Prediction
def get_valid_input(feature_name, min_val, max_val):
    while True:
        try:
            user_input = input(f"Enter {feature_name} in cm [{min_val:.1f} - {max_val:.1f}]: ")
            val = float(user_input)
            if val < min_val or val > max_val:
                print(f"⚠️ Value must be between {min_val:.1f} and {max_val:.1f}")
            else:
                return val
        except ValueError:
            print("⚠️ Please enter a valid number.")

print("\n🔮 Predict the Iris Flower Type")

sepal_length = get_valid_input('Sepal length', df['sepal length (cm)'].min(), df['sepal length (cm)'].max())
sepal_width = get_valid_input('Sepal width', df['sepal width (cm)'].min(), df['sepal width (cm)'].max())
petal_length = get_valid_input('Petal length', df['petal length (cm)'].min(), df['petal length (cm)'].max())
petal_width = get_valid_input('Petal width', df['petal width (cm)'].min(), df['petal width (cm)'].max())

user_features = [[sepal_length, sepal_width, petal_length, petal_width]]
user_prediction = model.predict(user_features)
predicted_species = iris.target_names[user_prediction][0]

print(f"\n🌸 The predicted Iris flower species is: **{predicted_species.capitalize()}**")
