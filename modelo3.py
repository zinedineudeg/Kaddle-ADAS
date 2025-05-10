import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.tree import plot_tree

df = pd.read_csv('winequality-red.csv')  # verifica el nombre exacto del archivo

print(df.head())
print(df.info())
print(df.describe())
print('\nValores de la calidad ',df['quality'].value_counts())


#vemos la distribución de la calidad del vino
sb.countplot(data=df, x='quality')
plt.title('Distribución de la calidad del vino')
plt.xlabel('Calidad')
plt.ylabel('Conteo')
plt.show()

#procesamos los datos 
df['quality_label'] = df['quality'].apply(lambda x: 1 if x >= 7 else 0)

X = df.drop(['quality', 'quality_label'], axis=1)
y = df['quality_label']

#Seleccionamos la calidad del vino
selector = SelectKBest(score_func=f_classif, k=8)
X_new = selector.fit_transform(X, y)
selected_features = X.columns[selector.get_support()]


X = df[selected_features]

#dividimos en entreno y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#entrenamos el modelo
modelo = DecisionTreeClassifier(max_depth=5, random_state=42)
modelo.fit(X_train, y_train)

#lo ponemos en prueba
y_pred = modelo.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
print("\nReporte de clasificación:\n", classification_report(y_test, y_pred))

#aca visualizamos los datos
plt.figure(figsize=(20, 10))
plot_tree(modelo, feature_names=X.columns, class_names=["Malo", "Bueno"], filled=True)
plt.title("Árbol de Decisión - Calidad del Vino")
plt.show()
