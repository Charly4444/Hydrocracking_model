from flask import Flask, render_template, request, jsonify
from flask_cors import CORS, cross_origin
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
import matplotlib.pyplot as plt
import os
import base64

app = Flask(__name__, static_folder='templates/static')

# added for CORS
CORS(app, resources={r"/*": {"origins": "*"}})

current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir,'..','hydro_predictor.h5')
a2cmodel_path = os.path.join(current_dir,'..','A2C_predictor')

scaler_path = os.path.join(current_dir,'..','scaler.pkl')

# get the scaler ready
scaler = joblib.load(scaler_path)

# load model
model = tf.keras.models.load_model(model_path)

# =======================================================================
# ===========Loading the A2C model is a bit more involved================
A2Cmodel = tf.saved_model.load(a2cmodel_path)
# =======================================================================

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
@cross_origin()
def predict():
    try:
        mode= int(request.form['mode'])
        submode_manual= int(request.form['submode_manual'])
        velo_data = float(request.form['velo_data'])
        temp_data = float(request.form['temp_data'])
        
        
        if (mode==0):
            # predict naphtha, diesel, kero
            new_data1 = np.array([[velo_data,temp_data]])
            new_scaled1 = scaler.transform(new_data1)
            predictions1 = model.predict(new_scaled1)
            
            
            # change to df & to percent values
            predictions_df = pd.DataFrame(predictions1, columns=['naphtha', 'diesel', 'kero'])
            predictions_df['naphtha'] = (abs(predictions_df['naphtha']/0.077))*100
            predictions_df['diesel'] = (abs(predictions_df['diesel']/0.680))*100
            predictions_df['kero'] = (abs(predictions_df['kero']/0.113))*100
            # back to np array
            percent_predictions = predictions_df[['naphtha','diesel','kero']].values
            
            # enlist the values
            return jsonify({'prediction': percent_predictions.tolist()[0]})
        
        elif (mode==1):
            if(submode_manual==0):
                # predict naphtha
                new_data1 = np.array([[velo_data,temp_data]])
                new_scaled1 = scaler.transform(new_data1)
                predictions1 = model.predict(new_scaled1)
                
                # change to df & to percent values
                predictions_df = pd.DataFrame(predictions1, columns=['naphtha', 'diesel', 'kero'])
                predictions_df['naphtha'] = (predictions_df['naphtha']/0.077)*100
                predictions_df['diesel'] = (predictions_df['diesel']/0.680)*100
                predictions_df['kero'] = (predictions_df['kero']/0.113)*100
        
                # back to np array
                percent_predictions = predictions_df[['naphtha','diesel','kero']].values
                
                # enlist the values
                return jsonify({'prediction': percent_predictions.tolist()[0]})
            
            elif(submode_manual==1):
                #  predict diesel
                new_data2 = np.array([[velo_data,temp_data]])
                new_scaled2 = scaler.transform(new_data2)
                predictions2 = model.predict(new_scaled2)
        
                # change to df & to percent values
                predictions_df = pd.DataFrame(predictions2, columns=['naphtha', 'diesel', 'kero'])
                predictions_df['naphtha'] = (predictions_df['naphtha']/0.077)*100
                predictions_df['diesel'] = (predictions_df['diesel']/0.680)*100
                predictions_df['kero'] = (predictions_df['kero']/0.113)*100

                # back to np array
                percent_predictions = predictions_df[['naphtha','diesel','kero']].values
        
                # enlist the values
                return jsonify({'prediction': percent_predictions.tolist()[0]})
                
            elif (submode_manual==2):
                # predict kero
                new_data3 = np.array([[velo_data,temp_data]])
                new_scaled3 = scaler.transform(new_data3)
                predictions3 = model.predict(new_scaled3)
        
                # change to df & to percent values
                predictions_df = pd.DataFrame(predictions3, columns=['naphtha', 'diesel', 'kero'])
                predictions_df['naphtha'] = (predictions_df['naphtha']/0.077)*100
                predictions_df['diesel'] = (predictions_df['diesel']/0.680)*100
                predictions_df['kero'] = (predictions_df['kero']/0.113)*100

                # back to np array
                percent_predictions = predictions_df[['naphtha','diesel','kero']].values
        
                # enlist the values
                return jsonify({'prediction': percent_predictions.tolist()[0]})
            
        elif(mode==2):
            print("I AM STILL WORKING ON THIS PART TO DISPLAY FROM THE RL ALG")
            current_values = np.array([velo_data, temp_data])
            initial_state = scaler.transform([current_values])
            # Convert the initial state to the correct data type (tf.int32) => crucial to not fail
            initial_state_tensor = tf.constant(initial_state.reshape(1, -1), dtype=tf.int32)
            # ===== The prediction process for A2c is a bit more involved too =====
            num_steps = 10
            for step in range(num_steps):
                actor_actions, _ = A2Cmodel(initial_state_tensor)
                action = np.random.normal(actor_actions[0], 0.1)
                current_values = current_values + action
            
            # now, predict products
            new_data4 = np.array([current_values])
            new_scaled4 = scaler.transform(new_data4)
            predictions4 = model.predict(new_scaled4)

            # change to df & to percent values
            predictions_df = pd.DataFrame(predictions4, columns=['naphtha', 'diesel', 'kero'])
            predictions_df['naphtha'] = (abs(predictions_df['naphtha']/0.077))*100
            predictions_df['diesel'] = (abs(predictions_df['diesel']/0.680))*100
            predictions_df['kero'] = (abs(predictions_df['kero']/0.113))*100

            # back to np array
            percent_predictions = predictions_df[['naphtha','diesel','kero']].values
            
            plt.bar(['Naphtha', 'Diesel', 'Kero'], percent_predictions[0])
            plt.xlabel('Products')
            plt.ylabel('Yield Percentage')
            plt.title('Predicted Yields')
            current_dir = os.path.dirname(os.path.realpath(__file__))
            image_path = os.path.join(current_dir, 'templates', 'static', 'bar_chart.png')
            plt.savefig(image_path)
            

            # Close the figure to release resources
            plt.close()

            # Encode the image to base64 for sending in the JSON response
            with open(image_path, 'rb') as image_file:
                encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
            return jsonify({
                'prediction': percent_predictions.tolist()[0],
                'encoded_image': encoded_image
            })

        
    except Exception as e:
        return jsonify({'error': str(e)})
    
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
    