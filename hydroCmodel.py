import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import joblib

# # # Load the modified dataset
# data = pd.read_csv('fulldata_m.csv')

# # Columns: [velocity, T(K), C_H, Naphtha, Diesel, Kero]

# # Extracting input features (X) and target variables (y)
# X = data[['velocity', 'T(K)']].values

# y = data[['Naphtha', 'Diesel', 'Kero']].values

# # Splitting the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Standardizing the input features
# scaler = MinMaxScaler()
# scaler.fit(X_train)

# # save the fitted scaler, for later use
# joblib.dump(scaler, 'scaler.pkl')

# X_train_scaled = scaler.transform(X_train)
# X_test_scaled = scaler.transform(X_test)

# # Building the neural network model
# model = tf.keras.Sequential([
#     tf.keras.layers.Dense(64, activation='relu', input_shape=(2,)),
#     tf.keras.layers.Dense(32, activation='relu'),
#     tf.keras.layers.Dense(3, activation='linear')  # Assuming 3 output
# ])

# # Compiling the model
# optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
# model.compile(optimizer=optimizer, loss='mean_squared_error')

# # Training the model
# history = model.fit(X_train_scaled, y_train, epochs=20, batch_size=32, validation_data=(X_test_scaled, y_test))


# # Visualizing the training and validation loss
# plt.plot(history.history['loss'], label='Training Loss')
# plt.plot(history.history['val_loss'], label='Validation Loss')
# plt.title('Training and Validation Loss')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.legend()
# plt.savefig("gen_IMAGES/H_trainloss.png")
# plt.show()


# #  Save the trained model
# model.save('hydro_predictor.h5')

# # Save the model weights
# model.save_weights('hydro_model_weights.h5')

# # ===============================================================================
# # Evaluate the model on the test set
# loss = model.evaluate(X_test_scaled, y_test)
# print(f"Test Loss: {loss}")

# ===============================================================================
# UPDATED FOR PREDICTIONS

# Make predictions on new data with immediate model
new_velocity = 230
new_temperature = 625

scaler = joblib.load('scaler.pkl')
model = tf.keras.models.load_model('hydro_predictor.h5')

new_data = np.array([[new_velocity, new_temperature]])
new_data_scaled = scaler.transform(new_data)
predictions = model.predict(new_data_scaled)

print(f"at v={new_velocity}, T={new_temperature}K, \n you will get these yields=> {predictions} \n for naphtha, diesel and kero respectively")
