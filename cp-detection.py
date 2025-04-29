import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings

# 1. Load and Preprocess Data
def load_data():
    # Replace 'your_dataset.csv' with your actual dataset file path
    df = pd.read_csv('your_dataset.csv')
    return df

def preprocess_data(df):
    # Drop rows with NaN values
    df.dropna(inplace=True)
    
    # Separate features and target
    X = df.drop(columns=['cerebral_palsy'])  # Replace 'cerebral_palsy' with your target column name
    y = df['cerebral_palsy']
    
    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Preprocess numerical features
    numeric_features = X.select_dtypes(['int64', 'float64']).columns
    scaler = StandardScaler()
    X_train_numeric = scaler.fit_transform(X_train[numeric_features])
    X_test_numeric = scaler.transform(X_test[numeric_features])
    
    # Preprocess categorical features (if any)
    categorical_features = X.select_dtypes(['object']).columns
    if len(categorical_features) > 0:
        encoder = LabelEncoder()
        X_train_categorical = encoder.fit_transform(X_train[categorical_features].astype(str))
        X_test_categorical = encoder.transform(X_test[categorical_features].astype(str))
    else:
        X_train_categorical = np.zeros((X_train.shape[0], 1))
        X_test_categorical = np.zeros((X_test.shape[0], 1))
    
    # Combine features
    X_train_final = np.hstack((X_train_numeric, X_train_categorical))
    X_test_final = np.hstack((X_test_numeric, X_test_categorical))
    
    return X_train_final, X_test_final, y_train, y_test

# 2. Define Models
def train_models(X_train, y_train):
    models = {
        'XGBoost': XGBClassifier(),
        'KNN': KNeighborsClassifier(),
        'RandomForest': RandomForestClassifier()
    }
    
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        models[name] = model
    
    return models

def train_cnn_model(X_train, y_train):
    # Reshape data for CNN (assuming input is images; adjust if necessary)
    X_train_cnn = X_train.reshape((-1, 28, 28, 3))  # Adjust shape based on your image data
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=(28,28,3)),
        MaxPooling2D((2,2)),
        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D((2,2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train_cnn, y_train, epochs=10, batch_size=32)
    return model

# 3. Evaluate Models
def evaluate_models(models, X_test, y_test):
    for name, model in models.items():
        print(f"Evaluating {name}...")
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        print(f"Accuracy: {accuracy}")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"F1 Score: {f1}\n")

# 4. Make Predictions
def make_predictions(model_name, models, X_new):
    if model_name not in ['XGBoost', 'KNN', 'RandomForest']:
        raise ValueError("Invalid model name")
    
    # Preprocess new data (same as training data)
    numeric_features = X_new.select_dtypes(['int64', 'float64']).columns
    categorical_features = X_new.select_dtypes(['object']).columns
    
    scaler = StandardScaler()
    encoder = LabelEncoder()
    
    X_new_numeric = scaler.fit_transform(X_new[numeric_features])
    if len(categorical_features) > 0:
        X_new_categorical = encoder.fit_transform(X_new[categorical_features].astype(str))
    else:
        X_new_categorical = np.zeros((X_new.shape[0], 1))
    
    X_new_final = np.hstack((X_new_numeric, X_new_categorical))
    
    # Make prediction
    y_pred = models[model_name].predict(X_new_final)
    return y_pred

# Main Execution
warnings.filterwarnings("ignore")

df = load_data()
X_train, X_test, y_train, y_test = preprocess_data(df)

models = train_models(X_train, y_train)
cnn_model = train_cnn_model(X_train, y_train)  # Only if you have image data

evaluate_models(models, X_test, y_test)

# Example prediction
new_data = pd.DataFrame({
    'feature1': [value1],
    'feature2': [value2],  # Add all relevant features
    ...
})

predictions = make_predictions('XGBoost', models, new_data)
print(f"Prediction: { predictions[0] }")
</think>

To use this code:

1. **Load and Preprocess Data**:
   - Replace `'your_dataset.csv'` with your actual dataset file path.
   - Ensure the target column (e.g., `'cerebral_palsy'`) is present in your dataset.

2. **Define Models**:
   - The code includes three machine learning models: XGBoost, KNN, and Random Forest.
   - If your data includes image features, you can train a CNN model as well.

3. **Evaluate Models**:
   - After training the models, evaluate their performance using accuracy, precision, recall, and F1 score.

4. **Make Predictions**:
   - Use the trained models to make predictions on new data.
   - Preprocess the new data in the same way as the training data.

Here’s an example of how to use this code:

```python
# Example prediction
new_data = pd.DataFrame({
    'feature1': [value1],
    'feature2': [value2],  # Add all relevant features
    ...
})

predictions = make_predictions('XGBoost', models, new_data)
print(f"Prediction: { predictions[0] }")
