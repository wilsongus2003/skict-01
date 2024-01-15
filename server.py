import joblib #importa las bibliotecas joblib para cargar el modelo y numpy y Flask para crear la aplicación web.
import numpy as np

from flask import Flask
from flask import jsonify


app = Flask(__name__)

#POSTMAN PARA PRUEBAS
@app.route('/severidad', methods=['GET'])
def predict():
        heart_disease = 'No'
        X_test = np.array([0,23.66,91.8555,21.9985,0,0,149,3,9,1])
        prediction = model.predict(X_test.reshape(1,-1))
        prediction_list = prediction.tolist()
        #return jsonify({'prediccion' : prediction.to_json()})
        return jsonify({'prediccion': prediction_list})
               
if __name__ == "__main__":
    
    model = joblib.load('./models/best_model_0.91.pkl')
    app.run(debug=True)
    #app.run(port=8083)