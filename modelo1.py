import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error


df = pd.read_csv('Car details v3.csv')
print(f'Primeras filas \n {df.head()}')
print(f'\n Info del archivo (que columnas y tipo de variables) \n {df.info()}')
print(f'\n Descripción de la información \n { df.describe()}')
print(f'\n mostramos la cantidad de valores que hay en null \n {df.isnull().sum()}')



#empezamos la limpieza de algunos datos
# Convertir variables categóricas a numéricas
df.drop_duplicates(inplace=True)
df.dropna(inplace=True)

# convertimos las variables categoricas a numericas 
df['fuel'] = df['fuel'].astype('category').cat.codes
df['seller_type'] = df['seller_type'].astype('category').cat.codes
df['transmission'] = df['transmission'].astype('category').cat.codes
df['owner'] = df['owner'].astype('category').cat.codes
df['mileage'] = df['mileage'].astype('category').cat.codes
df['engine'] = df['engine'].astype('category').cat.codes
df['max_power'] = df['max_power'].astype('category').cat.codes
df['torque'] = df['torque'].astype('category').cat.codes
df['brand'] = df['name'].apply(lambda x: x.split()[0])
df['brand'] = df['brand'].astype('category').cat.codes
df.drop('name', axis=1, inplace=True)

#Seleccionamos las variables
X = df.drop(['selling_price'], axis=1)
y = df['selling_price']

#separaramos en entrenamiento y prueba
x_train, x_test, y_train, y_test = train_test_split(X,y,test_size= 0.2, random_state= 42)

modelo = LinearRegression()
modelo.fit(x_train, y_train)

#prediccion
y_pred = modelo.predict(x_test)

print('\n Coeficientes del modelo \n')
for nombre, coef in zip(X.columns, modelo.coef_):
    print(f'{nombre},{coef}')


print('\nR2 Score',r2_score(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
print('Intercepto: ',modelo.intercept_)

plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y.min(),y.max()],[y.min(),y.max()])
plt.xlabel("Actual precio")
plt.ylabel("Precio predicho")
plt.title("Actual vs Precio predicho")
plt.grid(True)
plt.tight_layout()
plt.show()

