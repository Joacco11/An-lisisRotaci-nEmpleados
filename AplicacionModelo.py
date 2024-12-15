import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer

# Función para clasificar columnas en numéricas y categóricas
def classify_columns(csv_path):

    data = pd.read_csv(csv_path) 
    numeric_columns = data.select_dtypes(include=['number']).columns.tolist()
    categorical_columns = data.select_dtypes(exclude=['number']).columns.tolist()
    return data, numeric_columns, categorical_columns

# Ruta del archivo CSV
csv_path = "https://raw.githubusercontent.com/Joacco11/An-lisisRotaci-nEmpleados/refs/heads/master/HR-Employee-Attrition.csv"

# Cargar y clasificar columnas
new_data, numerics, categoricals = classify_columns(csv_path)

# Remover la columna 'Attrition' de las categóricas, si aplica
if 'Attrition' in categoricals:
    categoricals.remove('Attrition')

# Cargar el transformador y el modelo PCA desde los archivos guardados
loaded_transformer = joblib.load(r'C:/Users/User/Desktop/Sexto ciclo/ProyectoIntegrador/proyecto/DesarrolloModelo/data_transformer.pkl')
pca_model = joblib.load(r'C:/Users/User/Desktop/Sexto ciclo/ProyectoIntegrador/proyecto/DesarrolloModelo/pca_model.pkl')

# Obtener los nombres de las columnas después de OneHotEncoder
ohe_columns = loaded_transformer.named_transformers_['cat'].get_feature_names_out(categoricals)
column_names = numerics + ohe_columns.tolist()

# Transformar la nueva data utilizando el transformador cargado
new_data_transformed = loaded_transformer.transform(new_data)
new_data_transformed.shape
# Reducir la dimensionalidad con el modelo PCA
new_data_pca = pca_model.transform(new_data_transformed)

# Cargar el modelo XGBoost guardado
best_svc_model = joblib.load(r'C:/Users/User/Desktop/Sexto ciclo/ProyectoIntegrador/proyecto/DesarrolloModelo/best_svc_model.joblib')
# Realizar predicciones con los datos transformados
predicciones = best_svc_model.predict(new_data_transformed)

# Crear un DataFrame con las predicciones
pred_df = pd.DataFrame(predicciones, columns=['Predicción'])

# Mostrar las predicciones
print("Predicciones realizadas:")
print(pred_df.head())

# Guardar las predicciones en un archivo CSV
pred_df.to_csv('predicciones_despliegue_svc.csv', index=False)
print("Predicciones guardadas en 'predicciones_despliegue_svc.csv'")
