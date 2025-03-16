import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load the data
def load_data():
    try:
        train_data = pd.read_csv('train.csv')
        test_data = pd.read_csv('test.csv')
        return train_data, test_data
    except FileNotFoundError:
        print("Error: Please ensure train.csv and test.csv are in the current directory")
        return None, None

def preprocess_data(df):
    if df is None:
        return None
    
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Fill missing values
    data['Age'].fillna(data['Age'].median(), inplace=True)
    data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)
    data['Fare'].fillna(data['Fare'].median(), inplace=True)
    
    # Convert categorical features
    data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})
    data['Embarked'] = data['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
    
    # Create title feature from Name
    data['Title'] = data['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    title_mapping = {
        'Mr': 1, 'Miss': 2, 'Mrs': 3, 'Master': 4,
        'Dr': 5, 'Rev': 5, 'Col': 5, 'Major': 5, 'Mlle': 2,
        'Countess': 3, 'Ms': 2, 'Lady': 3, 'Jonkheer': 1,
        'Don': 1, 'Mme': 3, 'Capt': 5, 'Sir': 5
    }
    data['Title'] = data['Title'].map(title_mapping)
    data['Title'].fillna(0, inplace=True)
    
    # Select features for model
    features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Title']
    return data[features]

def train_model(X_train, y_train):
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Train Random Forest model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    return model, scaler

def main():
    # Load data
    train_data, test_data = load_data()
    if train_data is None or test_data is None:
        return
    
    # Preprocess data
    X = preprocess_data(train_data)
    y = train_data['Survived']
    
    # Split the data
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model, scaler = train_model(X_train, y_train)
    
    # Validate model
    X_val_scaled = scaler.transform(X_val)
    val_predictions = model.predict(X_val_scaled)
    
    # Print results
    print("\nModel Performance:")
    print(f"Validation Accuracy: {accuracy_score(y_val, val_predictions):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_val, val_predictions))
    
    # Create feature importance plot
    plt.figure(figsize=(10, 6))
    importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    sns.barplot(x='importance', y='feature', data=importance)
    plt.title('Feature Importance in Survival Prediction')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.close()

if __name__ == "__main__":
    main()
