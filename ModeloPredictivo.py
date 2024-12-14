import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn import tree
from sklearn.svm import SVC
import joblib
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns



data_final = pd.read_csv("C:\\Users\\User\\Desktop\\Sexto ciclo\\ProyectoIntegrador\\proyecto\\DesarrolloModelo\\data_final.csv")
data_final.head
X = data_final.drop(columns=['Attrition'])  
y = data_final['Attrition']
X.shape
y.shape


#Separación de la data
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size = 0.3, 
                                                    random_state = 1)








#Selección del modelo predictivo

#XGBoost
model = XGBClassifier(random_state=153468)
# Definir los hiperparámetros a ajustar
param_grid = {
    'n_estimators': [50, 100, 150],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 7, 5, 6],
    'min_child_weight': [1, 2, 3],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 0.9],
    'gamma': [0, 0.1, 0.2],
}

# Configurar el Grid Search
grid_search = GridSearchCV(estimator=model,
                           param_grid=param_grid,
                           scoring='accuracy',  
                           cv=3,  # Número de divisiones para la validación cruzada
                           verbose=1,
                           ) 
# Ajustar el modelo a los datos
grid_search.fit(X_train, y_train)

# Mejor combinación de parámetros
print("Mejores parámetros encontrados: ", grid_search.best_params_)

# Predecir con el mejor modelo
best_xgboost_model = grid_search.best_estimator_
y_pred = best_xgboost_model.predict(X_test)

# Calcular y mostrar la precisión
cnf_matrix_xgb = confusion_matrix(y_test, y_pred)
print("Matriz de Confusión (XGBoost):\n", cnf_matrix_xgb)

# Imprimir el informe de clasificación
print(classification_report(y_test, y_pred))
# Guardar el modelo de XGBoost
joblib.dump(best_xgboost_model, 'best_xgboost_model.joblib')
predictions_df = pd.DataFrame({
    'Predicciones': y_pred,
    'Verdadero': y_test  # Incluyendo los valores verdaderos para comparación
})

# Guardar las predicciones en un archivo CSV
predictions_df.to_csv('predicciones_xgboost.csv', index=False)
##--------------------------------------------------------------------------------------------------------------------------
#RandomForest

rforest_model = RandomForestClassifier(random_state=100)

param_grid_rf = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 'log2'],
}
grid_search_rf = GridSearchCV(estimator=RandomForestClassifier(random_state=100),
                               param_grid=param_grid_rf,
                               scoring='accuracy',
                               cv=3,  # Usar validación cruzada
                               verbose=1,
                               )
grid_search_rf.fit(X_train, y_train)
best_rf_model = grid_search_rf.best_estimator_
y_pred_rf = best_rf_model.predict(X_test)
# Evaluación del modelo con la matriz de confusión
cnf_matrix_rf = confusion_matrix(y_test, y_pred_rf)
print("Matriz de Confusión (RandomForest):\n", cnf_matrix_rf)
print(classification_report(y_test, y_pred_rf))

fig, axes = plt.subplots(nrows=4, ncols=5, figsize=(10, 8), dpi=900)
id = 0 
for j in range(0, 4):
    for index in range(0, 5):
        id += 1
        if id < len(best_rf_model.estimators_):  
            tree.plot_tree(best_rf_model.estimators_[id],
                           feature_names=X_train.columns, 
                           filled=True,
                           ax=axes[j, index])
            axes[j, index].set_title('Estimador: ' + str(id), fontsize=11)

fig.tight_layout()
fig.savefig('ModelosRandomForest.pdf')
# Guardar el modelo de Random Forest
joblib.dump(best_rf_model, 'best_random_forest_model.joblib')
predictions_df = pd.DataFrame({
    'Predicciones': y_pred_rf,
    'Verdadero': y_test  # Incluyendo los valores verdaderos para comparación
})

# Guardar las predicciones en un archivo CSV
predictions_df.to_csv('predicciones_rf.csv', index=False)
##------------------------------------------------------------------------------------------------------------------------
#Support Vector Classifier
svc_model = SVC(random_state=159)

param_grid_svc = {
    'C': [0.1, 1, 10, 100],  # Parámetro de regularización
    'kernel': ['linear', 'rbf', 'poly'],  # Tipo de kernel
    'gamma': ['scale', 'auto'] + [0.001, 0.01, 0.1, 1],  # Parámetro de decisión del kernel
}

# Configurar el Grid Search
grid_search_svc = GridSearchCV(estimator=svc_model,
                                 param_grid=param_grid_svc,
                                 scoring='accuracy',  # Medida de rendimiento
                                 cv=3,  # Número de divisiones para la validación cruzada
                                 verbose=1)
# Ajustar el modelo a los datos
grid_search_svc.fit(X_train, y_train)

# Mejor combinación de parámetros
print("Mejores parámetros encontrados: ", grid_search_svc.best_params_)


# Predecir con el mejor modelo
best_svc_model = grid_search_svc.best_estimator_
y_pred_svc = best_svc_model.predict(X_test)

# Calcular y mostrar la precisión
accuracy = accuracy_score(y_test, y_pred_svc)
print(f"Precisión: {accuracy}")

# Evaluación del modelo
cnf_matrix_svc = confusion_matrix(y_test, y_pred_svc)
print("Matriz de Confusión (svc):\n", cnf_matrix_rf)
print(classification_report(y_test, y_pred_svc))

joblib.dump(best_svc_model, 'best_svc_model.joblib')
predictions_df = pd.DataFrame({
    'Predicciones': y_pred_svc,
    'Verdadero': y_test  # Incluyendo los valores verdaderos para comparación
})

# Guardar las predicciones en un archivo CSV
predictions_df.to_csv('predicciones_svc.csv', index=False)
#-------------------------------------------------------------------------------------------
# Evaluación de cada modelo

# Diccionario para almacenar los resultados
results = {}

# Definir el diccionario de modelos y sus respectivas predicciones
models = {
    "XGBoost": (best_xgboost_model, y_pred),
    "Random Forest": (best_rf_model, y_pred_rf),
    "SVC": (best_svc_model, y_pred_svc)
}

# Calcular las métricas para cada modelo
for model_name, (model, predictions) in models.items():
    # Calcular las métricas
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    
    # Almacenar los resultados en el diccionario
    results[model_name] = {
        "Precisión": precision,
        "Recall": recall,
        "F1-Score": f1,
        "Exactitud": accuracy,
    }

# Convertir los resultados en un DataFrame
results_df = pd.DataFrame(results).T

# Mostrar el DataFrame con los resultados
print(results_df)