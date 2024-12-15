import subprocess
import sys
import os

def install_package(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Verificar si los paquetes están instalados, si no, instalarlos
try:
    from flask import Flask, request, jsonify
except ImportError:
    install_package("flask")

try:
    import pandas as pd
except ImportError:
    install_package("pandas")

try:
    import sklearn
except ImportError:
    install_package("scikit-learn")

try:
    import joblib
except ImportError:
    install_package("joblib")

print("Todos los paquetes están instalados.")

# Cargar transformador y modelo
transformer = joblib.load('data_transformer.pkl')
model = joblib.load('best_svc_model.joblib')

# Inicializar la aplicación Flask
app = Flask(__name__)

# Ruta para la API de ingesta individual
@app.route('/api/ingesta-individual', methods=['POST'])
def ingesta_individual():
    """
    API para la ingesta de datos individuales de empleados.
    """
    try:
        # Recibir datos del empleado en formato JSON
        data = request.json
        
        # Validar que todos los campos requeridos estén presentes
        required_fields = ["Age", "MonthlyIncome", "JobSatisfaction", "DistanceFromHome", "YearsAtCompany"]  # Completar según tu dataset
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            return jsonify({"error": f"Faltan los siguientes campos: {', '.join(missing_fields)}"}), 400
        
        # Convertir los datos a un DataFrame
        input_data = pd.DataFrame([data])
        
        # Validar que no haya valores nulos
        if input_data.isnull().sum().sum() > 0:
            return jsonify({"error": "Los datos contienen valores nulos."}), 400
        
        # Transformar los datos
        transformed_data = transformer.transform(input_data)
        
        # Generar predicción
        prediction = model.predict(transformed_data)[0]
        
        # Crear respuesta
        response = {
            "input_data": data,
            "prediction": int(prediction),  # 0: No abandona, 1: Abandona
            "message": "Predicción realizada con éxito."
        }
        
        # Almacenar en archivo CSV
        save_data(data, prediction)
        
        return jsonify(response), 200
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def save_data(data, prediction):
    """
    Guardar datos y predicción en un archivo CSV.
    """
    output_file = "ingesta_individual.csv"
    # Crear DataFrame con los datos y la predicción
    row = pd.DataFrame([{**data, "Prediction": prediction}])
    
    # Si el archivo ya existe, añadir sin sobrescribir
    if os.path.exists(output_file):
        row.to_csv(output_file, mode='a', header=False, index=False)
    else:
        row.to_csv(output_file, mode='w', header=True, index=False)

# Ejecutar la aplicación
if __name__ == "__main__":
    app.run(debug=True)


