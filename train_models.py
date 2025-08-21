import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import joblib
import json
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os

def create_sequences(data, labels, dates, sequence_length=10):
    X, y, seq_dates = [], [], []
    
    sorted_indices = np.argsort(dates)
    data_sorted = data.iloc[sorted_indices]
    labels_sorted = labels.iloc[sorted_indices]
    dates_sorted = dates.iloc[sorted_indices]
    
    for i in range(len(data_sorted) - sequence_length):
        date_diff = (dates_sorted.iloc[i + sequence_length] - dates_sorted.iloc[i]).days
        if date_diff <= sequence_length + 5:
            X.append(data_sorted.iloc[i:(i + sequence_length)].values)
            y.append(labels_sorted.iloc[i + sequence_length])
            seq_dates.append(dates_sorted.iloc[i + sequence_length])
    
    return np.array(X), np.array(y), pd.Series(seq_dates)

def create_forecast_sequences(data, sequence_length=10):
    X = []
    for i in range(len(data) - sequence_length + 1):
        X.append(data[i:(i + sequence_length)])
    return np.array(X)

df = pd.read_csv('data/weatherAUS.csv')
df = df[df['Location'] == 'Sydney']
df = df.dropna(subset=['RainTomorrow'])
df['RainTomorrow'] = df['RainTomorrow'].map({'Yes': 1, 'No': 0})
df['Date'] = pd.to_datetime(df['Date'])

df = df.sort_values('Date').reset_index(drop=True)

all_features = ['MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation', 'Sunshine', 
                'WindGustSpeed', 'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am', 
                'Humidity3pm', 'Pressure9am', 'Pressure3pm', 'Cloud9am', 'Cloud3pm', 
                'Temp9am', 'Temp3pm']

df[all_features] = df[all_features].fillna(df[all_features].mean())

X = df[all_features]
y = df['RainTomorrow']
dates = df['Date']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=all_features, index=df.index)

sequence_length = 10
X_seq, y_seq, dates_seq = create_sequences(X_scaled_df, y, dates, sequence_length)

if len(X_seq) > 100:
    rain_ratio = np.mean(y_seq)
    
    X_seq_train, X_seq_test, y_seq_train, y_seq_test, dates_train, dates_test = train_test_split(
        X_seq, y_seq, dates_seq, test_size=0.2, random_state=42, stratify=y_seq
    )
    
    from sklearn.utils.class_weight import compute_class_weight
    class_weights = compute_class_weight('balanced', classes=np.unique(y_seq_train), y=y_seq_train)
    class_weight_dict = {0: class_weights[0], 1: class_weights[1]}
    
    lstm_model = Sequential([
        Input(shape=(sequence_length, len(all_features))),
        LSTM(128, return_sequences=True),
        Dropout(0.3),
        LSTM(64, return_sequences=True),
        Dropout(0.3),
        LSTM(32, return_sequences=False),
        Dropout(0.3),
        Dense(16, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])
    
    lstm_model.compile(
        optimizer='adam', 
        loss='binary_crossentropy', 
        metrics=['accuracy', 'precision', 'recall']
    )
    
    lstm_model.summary()
    
    early_stopping = EarlyStopping(
        monitor='val_loss', 
        patience=15, 
        restore_best_weights=True
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss', 
        factor=0.5, 
        patience=10, 
        min_lr=0.0001
    )
    
    history = lstm_model.fit(
        X_seq_train, y_seq_train,
        epochs=50,
        batch_size=16,
        validation_data=(X_seq_test, y_seq_test),
        callbacks=[early_stopping, reduce_lr],
        class_weight=class_weight_dict,
        verbose=1
    )
    
    lstm_pred_prob = lstm_model.predict(X_seq_test, verbose=0)
    
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
    best_threshold = 0.5
    best_f1_rain = 0
    
    for threshold in thresholds:
        pred_temp = (lstm_pred_prob > threshold).astype(int)
        from sklearn.metrics import f1_score, precision_score, recall_score
        
        f1_rain = f1_score(y_seq_test, pred_temp, pos_label=1)
        precision_rain = precision_score(y_seq_test, pred_temp, pos_label=1, zero_division=0)
        recall_rain = recall_score(y_seq_test, pred_temp, pos_label=1)
        
        print(f"Threshold {threshold}: F1={f1_rain:.3f}, Precision={precision_rain:.3f}, Recall={recall_rain:.3f}")
        
        if f1_rain > best_f1_rain:
            best_f1_rain = f1_rain
            best_threshold = threshold
    
    print(f"Best threshold: {best_threshold} (F1-score for rain: {best_f1_rain:.3f})")
    
    lstm_pred = (lstm_pred_prob > best_threshold).astype(int)
    lstm_score = accuracy_score(y_seq_test, lstm_pred)
    
    print("="*60)
    print(f"LSTM Model Accuracy: {lstm_score:.4f}")
    print(f"Best Rain Detection Threshold: {best_threshold}")
    print("="*60)
    
    print("\nClassification Report:")
    print(classification_report(y_seq_test, lstm_pred, target_names=['No Rain', 'Rain']))
    
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 3, 1)
    plt.plot(history.history['loss'], label='Training Loss', linewidth=2, color='red')
    plt.plot(history.history['val_loss'], label='Validation Loss', linewidth=2, color='orange')
    plt.title('Model Loss', fontsize=12, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Binary Crossentropy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 3, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy', linewidth=2, color='green')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2, color='blue')
    plt.title('Model Accuracy', fontsize=12, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 3, 3)
    cm = confusion_matrix(y_seq_test, lstm_pred)
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix', fontweight='bold')
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ['No Rain', 'Rain'])
    plt.yticks(tick_marks, ['No Rain', 'Rain'])
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    plt.subplot(2, 3, 4)
    plt.hist(lstm_pred_prob, bins=20, alpha=0.7, color='purple')
    plt.axvline(x=0.5, color='red', linestyle='--', label='Decision Threshold')
    plt.title('Prediction Confidence Distribution')
    plt.xlabel('Predicted Probability')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 3, 5)
    recent_indices = np.argsort(dates_test)[-50:]
    recent_dates = dates_test.iloc[recent_indices]
    recent_actual = y_seq_test[recent_indices]
    recent_pred = lstm_pred[recent_indices]
    
    plt.scatter(recent_dates, recent_actual, alpha=0.6, label='Actual', color='blue', s=30)
    plt.scatter(recent_dates, recent_pred, alpha=0.6, label='Predicted', color='red', s=30, marker='x')
    plt.title('Recent Predictions vs Actual')
    plt.xlabel('Date')
    plt.ylabel('Rain Tomorrow (0=No, 1=Yes)')
    plt.legend()
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 3, 6)
    feature_importance = []
    for i, feature in enumerate(all_features):
        feature_data = X_scaled_df[feature].values
        correlation = np.abs(np.corrcoef(feature_data[sequence_length:len(y)], y[sequence_length:])[0,1])
        if np.isnan(correlation):
            correlation = 0
        feature_importance.append(correlation)
    
    sorted_indices = np.argsort(feature_importance)[::-1]
    top_10_features = [all_features[i] for i in sorted_indices[:10]]
    top_10_scores = [feature_importance[i] for i in sorted_indices[:10]]
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(top_10_features)))
    bars = plt.bar(range(len(top_10_features)), top_10_scores, color=colors)
    plt.title('Top 10 Feature Correlations\nwith Rain Target')
    plt.xlabel('Features')
    plt.ylabel('Correlation Score')
    feature_names_short = [f[:8] for f in top_10_features]
    plt.xticks(range(len(top_10_features)), feature_names_short, rotation=45, ha='right')
    plt.grid(True, alpha=0.3)
    
    for bar, score in zip(bars, top_10_scores):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, 
                f'{score:.3f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    os.makedirs('static', exist_ok=True)
    plt.savefig('static/lstm_comprehensive_analysis.png', bbox_inches='tight', dpi=300)
    plt.close()
    
    latest_sequence = X_scaled[-sequence_length:]
    forecast_days = 7
    future_predictions = []
    current_sequence = latest_sequence.copy()
    
    for day in range(forecast_days):
        pred_input = current_sequence.reshape(1, sequence_length, len(all_features))
        
        pred_prob = lstm_model.predict(pred_input, verbose=0)[0][0]
        pred_class = 1 if pred_prob > best_threshold else 0
        future_predictions.append({
            'day': day + 1,
            'probability': pred_prob,
            'prediction': 'Rain' if pred_class == 1 else 'No Rain',
            'confidence': pred_prob if pred_class == 1 else (1 - pred_prob)
        })
        
        next_day_features = np.mean(current_sequence[-3:], axis=0)
        current_sequence = np.vstack([current_sequence[1:], next_day_features])
    
    print("7-Day Weather Forecast:")
    print("="*50)
    for pred in future_predictions:
        date = df['Date'].max() + timedelta(days=pred['day'])
        print(f"Day {pred['day']} ({date.strftime('%Y-%m-%d')}): {pred['prediction']} "
               f"(Confidence: {pred['confidence']:.1%})")
    print("="*50)
    
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    historical_days = list(range(-30, 0))
    historical_rain = df['RainTomorrow'].tail(30).values
    future_days = [pred['day'] for pred in future_predictions]
    future_rain_prob = [pred['probability'] for pred in future_predictions]
    
    plt.plot(historical_days, historical_rain, 'o-', label='Historical (Actual)', color='blue', linewidth=2)
    plt.plot(future_days, future_rain_prob, 's-', label='Forecast (Probability)', color='red', linewidth=2)
    plt.axvline(x=0, color='black', linestyle='--', label='Today')
    plt.axhline(y=0.5, color='gray', linestyle=':', alpha=0.7, label='Rain Threshold')
    plt.title('Weather Forecast: Historical vs Predicted Rain Probability', fontweight='bold')
    plt.xlabel('Days (Negative = Past, Positive = Future)')
    plt.ylabel('Rain Probability')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 1, 2)
    colors = ['green' if pred['confidence'] > 0.7 else 'orange' if pred['confidence'] > 0.5 else 'red' 
              for pred in future_predictions]
    bars = plt.bar(future_days, [pred['confidence'] for pred in future_predictions], 
                   color=colors, alpha=0.7)
    plt.title('Forecast Confidence Levels', fontweight='bold')
    plt.xlabel('Days Ahead')
    plt.ylabel('Confidence Level')
    plt.ylim(0, 1)
    
    for bar, pred in zip(bars, future_predictions):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                f'{pred["confidence"]:.1%}', ha='center', va='bottom', fontweight='bold')
    
    plt.axhline(y=0.7, color='green', linestyle='-', alpha=0.3, label='High Confidence (>70%)')
    plt.axhline(y=0.5, color='orange', linestyle='-', alpha=0.3, label='Medium Confidence (50-70%)')
    plt.axhline(y=0.3, color='red', linestyle='-', alpha=0.3, label='Low Confidence (<50%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('static/lstm_weather_forecast.png', bbox_inches='tight', dpi=300)
    plt.close()
    
    os.makedirs('models', exist_ok=True)
    lstm_model.save('models/lstm_weather_model.keras')
    joblib.dump(scaler, 'models/lstm_scaler.joblib')
    
    forecast_data = {
        'model_accuracy': float(lstm_score),
        'training_epochs': len(history.history['loss']),
        'sequence_length': sequence_length,
        'features_used': all_features,
        'forecast_date': df['Date'].max().isoformat(),
        'predictions': future_predictions
    }
    
    with open('models/weather_forecast.json', 'w') as f:
        json.dump(forecast_data, f, indent=2, default=str)
    
    print("Model and forecast data saved successfully!")

else:
    print("Not enough data to create sequences. Need more historical weather data.")

def predict_tomorrow(weather_data):
    try:
        import tensorflow as tf
        model = tf.keras.models.load_model('models/lstm_weather_model.keras')
        scaler = joblib.load('models/lstm_scaler.joblib')
        
        scaled_data = scaler.transform(weather_data)
        
        prediction = model.predict(scaled_data.reshape(1, sequence_length, len(all_features)))
        probability = prediction[0][0]
        
        return {
            'probability': probability,
            'prediction': 'Rain' if probability > 0.5 else 'No Rain',
            'confidence': probability if probability > 0.5 else (1 - probability)
        }
    except Exception as e:
        print(f"Prediction error: {e}")
        return None

print("LSTM Weather Prediction System Complete!")