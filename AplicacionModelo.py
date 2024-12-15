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
loaded_transformer = joblib.load(r'C:/Users/User/Desktop/Sexto ciclo/ProyectoIntegrador/proyecto/data_transformer.pkl')
pca_model = joblib.load(r'C:/Users/User/Desktop/Sexto ciclo/ProyectoIntegrador/proyecto/pca_model.pkl')

# Obtener los nombres de las columnas después de OneHotEncoder
ohe_columns = loaded_transformer.named_transformers_['cat'].get_feature_names_out(categoricals)
column_names = numerics + ohe_columns.tolist()

# Transformar la nueva data utilizando el transformador cargado
new_data_transformed = loaded_transformer.transform(new_data)
new_data_transformed.shape
# Reducir la dimensionalidad con el modelo PCA
new_data_pca = pca_model.transform(new_data_transformed)

# Cargar el modelo XGBoost guardado
best_svc_model = joblib.load(r'C:/Users/User/Desktop/Sexto ciclo/ProyectoIntegrador/proyecto/best_svc_model.joblib')
# Realizar predicciones con los datos transformados
predicciones = best_svc_model.predict(new_data_transformed)

# Crear un DataFrame con las predicciones
pred_df = pd.DataFrame(predicciones, columns=['Predicción'])

# Combinar los datos originales con las predicciones
output_df = pd.concat([new_data.reset_index(drop=True), pred_df], axis=1)

# Mostrar las primeras filas del DataFrame combinado
print("Datos con predicciones:")
print(output_df.head())

# Guardar las predicciones junto con los datos originales en un archivo CSV
output_csv_path = 'predicciones_despliegue_svc_con_datos.csv'
output_df.to_csv(output_csv_path, index=False)

print(f"Predicciones guardadas en '{output_csv_path}'")
