import joblib #importa las bibliotecas joblib para cargar el modelo y numpy y Flask para crear la aplicación web.
import numpy as np

from flask import Flask
from flask import jsonify


app = Flask(__name__)

#POSTMAN PARA PRUEBAS
@app.route('/severidad', methods=['GET'])
def predict():
        heart_disease = 'No'
        X_test = np.array([0,23.5205,92.3315,21.9525,0,0,157,6])
        prediction = model.predict(X_test.reshape(1,-1))
        return jsonify({'prediccion' : prediction.tolist()})
        '''if prediction[0] == 1:
            heart_disease = 'Yes'
            return jsonify({'Tiene' : 1})
        else:
            return jsonify({'NO tiene' : 0})'''
        
if __name__ == "__main__":
    model = joblib.load('./models/best_model_0.6344.pkl')
    app.run(port=8083)