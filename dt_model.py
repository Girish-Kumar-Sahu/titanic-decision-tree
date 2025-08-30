"""
Titanic Survival Prediction using Decision Trees

Dataset: https://sololearn.com/uploads/files/titanic.csv
Author: Girish Kumar Sahu

This script builds a Decision Tree classifier on the Titanic dataset,
evaluates it using cross-validation, and visualizes the final tree.
"""

# 📦 Import libraries
import pandas as pd
import numpy as np
from sklearn.tree import export_graphviz, DecisionTreeClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score
import graphviz


# 🛠️ Step 1: Load and preprocess dataset
def load_data():
    """Load Titanic dataset and preprocess features."""
    df = pd.read_csv('https://sololearn.com/uploads/files/titanic.csv')

    # Feature engineering: create 'male' binary column
    df['male'] = df['Sex'] == 'male'

    # Select features and target
    features = ['Pclass', 'male', 'Age', 'Siblings/Spouses', 'Parents/Children', 'Fare']
    X = df[features].values
    y = df['Survived'].values

    return X, y, features


# 🛠️ Step 2: Model evaluation with cross-validation
def evaluate_decision_tree(X, y, criterion='gini', k=5):
    """
    Evaluate Decision Tree classifier using K-Fold cross-validation.

    Args:
        X: feature matrix
        y: target labels
        criterion: splitting criterion ('gini' or 'entropy')
        k: number of folds for cross-validation

    Returns:
        avg_accuracy, avg_precision, avg_recall, trained_model
    """
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    accuracy, precision, recall = [], [], []

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        dt = DecisionTreeClassifier(criterion=criterion, random_state=42)
        dt.fit(X_train, y_train)
        y_pred = dt.predict(X_test)

        accuracy.append(accuracy_score(y_test, y_pred))
        precision.append(precision_score(y_test, y_pred))
        recall.append(recall_score(y_test, y_pred))

    return np.mean(accuracy), np.mean(precision), np.mean(recall), dt


# 🛠️ Step 3: Export final tree
def export_tree(dt, features, filename="tree"):
    """Export trained decision tree as PNG using Graphviz."""
    dot_file = export_graphviz(
        dt,
        feature_names=features,
        class_names=['Not Survived', 'Survived'],
        filled=True, rounded=True, special_characters=True
    )
    graph = graphviz.Source(dot_file)
    graph.render(filename, format="png", cleanup=True)
    print(f"Decision tree saved as {filename}.png")


# 🏁 Main execution
if __name__ == "__main__":
    X, y, features = load_data()

    for criterion in ['gini', 'entropy']:
        acc, prec, rec, dt = evaluate_decision_tree(X, y, criterion)
        print(f"Decision Tree - {criterion}")
        print(f"Accuracy: {acc:.3f}")
        print(f"Precision: {prec:.3f}")
        print(f"Recall: {rec:.3f}\n")

    # Export the final trained tree (from last run)
    export_tree(dt, features)
