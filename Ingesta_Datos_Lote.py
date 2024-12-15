#Ejecutar en el terminal
#.\venv\Scripts\Activate
#pip install flask pandas scikit-learn joblib

from flask import Flask, request, jsonify
import pandas as pd
import joblib
import os
from datetime import datetime
import logging

# Configuración de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Cargar transformador y modelo preentrenado
transformer = joblib.load('data_transformer.pkl')
model = joblib.load('best_svc_model.joblib')

# Inicializar la aplicación Flask
app = Flask(__name__, root_path=os.path.dirname(os.path.abspath(__file__)))

# Ruta para la API de ingesta por lote
@app.route('/api/ingesta-lote', methods=['POST'])
def ingesta_lote():
    """
    API para la ingesta por lotes de datos de empleados.
    Recibe un archivo CSV, valida los datos, aplica el transformador y genera predicciones.
    """
    try:
        # Validar que se haya enviado un archivo
        if 'file' not in request.files:
            logging.error("No se ha enviado ningún archivo en la solicitud.")
            return jsonify({"error": "No se ha enviado ningún archivo."}), 400
        
        # Obtener el archivo enviado
        file = request.files['file']
        
        # Validar que el archivo tenga la extensión CSV
        if not file.filename.endswith('.csv'):
            logging.error(f"El archivo enviado no es un CSV: {file.filename}")
            return jsonify({"error": "El archivo debe ser un CSV."}), 400
        
        # Leer el archivo CSV en un DataFrame
        batch_data = pd.read_csv(file)
        logging.info(f"Archivo recibido: {file.filename}")
        logging.info(f"Columnas detectadas en el archivo: {list(batch_data.columns)}")
        
        # Validar que las columnas esperadas estén presentes en el archivo
        expected_columns = ["Age", "MonthlyIncome", "JobSatisfaction", "DistanceFromHome", "YearsAtCompany"]
        if not all(col in batch_data.columns for col in expected_columns):
            logging.error("El archivo no contiene todas las columnas requeridas.")
            return jsonify({"error": f"El archivo debe contener las columnas: {', '.join(expected_columns)}"}), 400
        
        # Validar que no haya valores nulos
        if batch_data.isnull().sum().sum() > 0:
            logging.error("El archivo contiene valores nulos.")
            return jsonify({"error": "El archivo contiene valores nulos."}), 400
        
        # Transformar los datos
        logging.info("Aplicando el transformador a los datos...")
        transformed_data = transformer.transform(batch_data)
        
        # Generar predicciones
        logging.info("Generando predicciones con el modelo preentrenado...")
        predictions = model.predict(transformed_data)
        
        # Añadir las predicciones al DataFrame original
        batch_data['Prediction'] = predictions
        
        # Guardar las predicciones en un archivo CSV con nombre dinámico
        output_file = save_predictions(batch_data)
        logging.info(f"Predicciones guardadas en el archivo: {output_file}")
        
        return jsonify({
            "message": "Predicciones realizadas con éxito.",
            "output_file": output_file,
            "head": batch_data.head().to_dict(orient="records")  # Muestra las primeras filas como ejemplo
        }), 200
    
    except Exception as e:
        logging.error(f"Error durante el procesamiento: {str(e)}")
        return jsonify({"error": str(e)}), 500

def save_predictions(data):
    """
    Guarda las predicciones en un archivo CSV con un nombre dinámico.
    :param data: DataFrame con las predicciones.
    :return: Nombre del archivo guardado.
    """
    # Crear un nombre único para el archivo usando la fecha y hora actual
    output_file = f"predicciones_lote_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    data.to_csv(output_file, index=False)
    return output_file

# Ejecutar la aplicación
if __name__ == "__main__":
    app.run(debug=True)
