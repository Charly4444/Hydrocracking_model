import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import joblib

# load model
model = tf.keras.models.load_model('hydro_predictor.h5')

# Load the modified dataset, write percent values
data = pd.read_csv('fulldata_m.csv')

# shuffle data & extract some
shuffled_data = data.sample(frac=1)
X_subset = shuffled_data[['velocity', 'T(K)']].iloc[:1000, :].values

# load the scaler and prepare data
scaler = joblib.load('scaler.pkl')

X_subset_scaled = scaler.transform(X_subset)

# Making predictions using the trained model
predictions = model.predict(X_subset_scaled)

# predictions to df to percent
input_df = pd.DataFrame(X_subset, columns=['velocity', 'T(K)'])

results_df = pd.DataFrame(predictions, columns=['naphtha', 'diesel', 'kero'])

results_df['naphtha'] = abs(results_df['naphtha'])
results_df['diesel'] = abs(results_df['diesel'])
results_df['kero'] = abs(results_df['kero'])

predictions_df = pd.concat([input_df, results_df], axis=1)

predictions_df['naphtha(%)'] = (predictions_df.apply(lambda row: row['naphtha']/0.077, axis=1))*100
predictions_df['diesel(%)'] = (predictions_df.apply(lambda row: row['diesel']/0.680, axis=1))*100
predictions_df['kero(%)'] = (predictions_df.apply(lambda row: row['kero']/0.113, axis=1))*100

# back to np array
percent_predictions = predictions_df[['naphtha(%)', 'diesel(%)', 'kero(%)']].values


# Extracting actual values for the first 1000 rows
actual_values = shuffled_data[['Naphtha(%)', 'Diesel(%)', 'Kero(%)']].iloc[:1000, :].values


# ==========   ================   =================   ===============
# Plotting the results for each product
products = ['Naphtha', 'Diesel', 'Kerosene']

for i in range(3):  # Assuming 3 products
    plt.figure(figsize=(10, 6))
    plt.plot(percent_predictions[:, i], label='Predicted')
    plt.plot(actual_values[:, i], label='Actual')
    plt.title(f'Prediction vs. Actual for {products[i]} 1000 samples')
    plt.xlabel('Sample Index')
    plt.ylabel('Yield(%)')
    plt.legend()
    plt.savefig(f"gen_IMAGES/{products[i]}_preds.png")
    plt.show()
