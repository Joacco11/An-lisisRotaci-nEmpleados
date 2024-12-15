import pandas as pd
import joblib
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA


def classify_columns(csv_path):
    # Cargar el archivo CSV
    data = pd.read_csv(csv_path) 
    # Identificar variables numéricas y categóricas
    numeric_columns = data.select_dtypes(include=['number']).columns.tolist()
    categorical_columns = data.select_dtypes(exclude=['number']).columns.tolist()

    return data, numeric_columns, categorical_columns

csv_path = "C:\\Users\\User\\Desktop\\Sexto ciclo\\ProyectoIntegrador\\proyecto\\DesarrolloModelo\\HR-Employee-Attrition.csv"
df, numerics, categoricals = classify_columns(csv_path)


num_var = ["Age", "AnnualIncome", "FamilyMembers", "ChronicDiseases"]
cat_var = ["Employment Type", "GraduateOrNot", "FrequentFlyer", "EverTravelledAbroad"]

ohe_columns = loaded_transformer.named_transformers_['cat'].get_feature_names_out(cat_var)
column_names = num_var + ohe_columns.tolist()

#Importar nueva data
loaded_transformer = joblib.load('data_transformer.pkl')
new_data_transformed = loaded_transformer.transform(new_data)
new_data_final = pd.DataFrame(new_data_transformed, columns=column_names)
print(new_data_final.head())


best_xgboost_model = joblib.load('best_xgboost_model.joblib')

# Realizar la predicción con la data transformada y reducida por PCA
predicciones = best_xgboost_model.predict(new_data_final)

# Presentar solo la predicción
pred_df = pd.DataFrame(predicciones, columns=['Predicción'])
print("Predicciones realizadas:")
print(pred_df)

# Si deseas guardar las predicciones en un archivo CSV
pred_df.to_csv('predicciones_despliegue_xgboost.csv', index=False)
print("Predicciones guardadas en 'PREDICCIÓNFINAL_predicciones_despliegue_xgboost.csv'")

