import pandas as pd
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
import joblib


import pandas as pd

def classify_columns(csv_path):
    # Cargar el archivo CSV
    data = pd.read_csv(csv_path) 
    # Identificar variables numéricas y categóricas
    numeric_columns = data.select_dtypes(include=['number']).columns.tolist()
    categorical_columns = data.select_dtypes(exclude=['number']).columns.tolist()

    return data, numeric_columns, categorical_columns

# Ruta del archivo CSV
csv_path = "C:\\Users\\User\\Desktop\\Sexto ciclo\\ProyectoIntegrador\\proyecto\\DesarrolloModelo\\HR-Employee-Attrition.csv"
df, numerics, categoricals = classify_columns(csv_path)

if 'Attrition' in categoricals:
    categoricals.remove('Attrition')

# Binarizar la columna 'Attrition'
label_encoder = LabelEncoder()
df['Attrition'] = label_encoder.fit_transform(df['Attrition'])

num_var = numerics
cat_var = categoricals
y = df['Attrition']

print("Variables numéricas:", num_var)
print("Variables categóricas:", cat_var)
print("Variables target:",y)

# Crear un preprocesador
preprocessor = ColumnTransformer(
    transformers=[
        ('num', MinMaxScaler(), num_var),  # Normalizar numéricas
        ('cat', OneHotEncoder(), cat_var)   # One-Hot Encoding para categóricas
    ]
)

# Ajustar y transformar las variables de entrada
X_transformed = preprocessor.fit_transform(df)

# Aplicar PCA

pca = PCA(n_components=30)  
X_pca = pca.fit_transform(X_transformed) 
sum(pca.explained_variance_ratio_)
X_pca.shape


# Guardar el transformador
joblib.dump(preprocessor, 'data_transformer.pkl')
joblib.dump(pca, 'pca_model.pkl')

# Obtener los nombres de las columnas resultantes y crear el dataframe
ohe_columns = preprocessor.named_transformers_['cat'].get_feature_names_out(cat_var)
column_names = num_var + ohe_columns.tolist()
# Crear el DataFrame final
X_final = pd.DataFrame(X_transformed, columns=column_names)
# Añadir el target al DataFrame final
X_final['Attrition'] = y.values
X_final.shape
# Mostrar las primeras filas del DataFrame final
print(X_final.head())

X_final.to_csv("data_final.csv", index=False, quoting=0)


#---------------------------------------------------------------------------------------------------------------------------

# (Opcional) Cargar el transformador y aplicarlo a un nuevo conjunto de datos
loaded_transformer = joblib.load('data_transformer.pkl')
new_data = pd.read_csv("ruta_del_nuevo_dataset.csv", sep=";")  # Cambia la ruta según sea necesario
new_data_transformed = loaded_transformer.transform(new_data)
new_data_final = pd.DataFrame(new_data_transformed, columns=column_names)
print(new_data_final.head())
