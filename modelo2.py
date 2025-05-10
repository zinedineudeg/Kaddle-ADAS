import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report



df = pd.read_csv('heart_cleveland_upload.csv')
print(f'Primeras filas \n {df.head()}')
print(f'\n Info del archivo (que columnas y tipo de variables) \n {df.info()}')
print(f'\n Descripción de la información \n { df.describe()}')
print(f'\n mostramos la cantidad de valores que hay en null \n{df.isnull().sum()}')


sb.countplot(data=df, x='condition')
plt.title("Distribución de presencia de enfermedad cardíaca")
plt.xlabel('Condicción')
plt.ylabel('Conteo')
plt.show()

# Correlación entre variables
sb.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Matriz de correlación")
plt.show()
#Eliminamos los valores nuloes si los hay
df.dropna(inplace=True) 

# Separar variables
X = df.drop('condition', axis=1)
y = df['condition']

#seleccionamos las caracteristicas
selector = SelectKBest(score_func=chi2, k=10)
X_new = selector.fit_transform(X, y)

caracteristicas = X.columns[selector.get_support()]
X = df[caracteristicas]

#divido en train y rtest
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#entrenamos el modelo
modelo = LogisticRegression(max_iter=1000)
modelo.fit(X_train, y_train)

y_pred = modelo.predict(X_test)

print("\nExactitud:", accuracy_score(y_test, y_pred))
print("Precisión:", precision_score(y_test, y_pred))
print("Sensibilidad:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
print("\nReporte de clasificación:\n", classification_report(y_test, y_pred))


conf_matrix = confusion_matrix(y_test, y_pred)

sb.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicción')
plt.ylabel('Actual')
plt.title('Matriz de Confusión')
plt.show()