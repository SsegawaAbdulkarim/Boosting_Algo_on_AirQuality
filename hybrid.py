from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Assume X_train, y_train, X_test, y_test are your training and testing data

# Define base models
base_models = [
    ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
    ('lr', LogisticRegression(random_state=42)),
    ('mlp', MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, random_state=42))
]

# Define meta-model
meta_model = MLPClassifier(hidden_layer_sizes=(50,), max_iter=1000, random_state=42)

# Create the stacked model
stacked_model = StackingClassifier(estimators=base_models, final_estimator=meta_model)

# Train the stacked model
stacked_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = stacked_model.predict(X_test)

# Evaluate the performance
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')
