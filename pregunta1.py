# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 01:19:53 2023

@author: Andres
"""

import csv
import math
import random

#/////////////////////--------CARGAR DATA SET IRIS----------///////////////////
with open('D:\INF-354\segunda_parte\Iris.csv', newline='') as dataset:
    reader = csv.reader(dataset)
    tabla = list(reader)
dataset.close()

for i in range(10):
    print(tabla[i])
#//////////////////////////////////////////////////////////////////////////////


#////////////////-----------SEPARAR NUMEROS Y CARACTERES-------////////////////
#separacion: 
#datos en x
#etiquetas y
X = [list(map(float, tabla_iris[1:5])) for tabla_iris in tabla[1:]]
y = [tabla_iris[5] for tabla_iris in tabla[1:]]

# Mostrar 10 tabla_iriss
print("datos flotante x")
for i in range(10):
    print(X[i])

print("\nnombre etiqueta y")
for i in range(10):
    print(y[i])
#//////////////////////////////////////////////////////////////////////////////


    
#//////////----------MEDIA Y DESVIACION ESTANDAR PARA ESTANDARIZAR X----------////////
num_filas = len(X)
num_columnas = len(X[0])

medias = list()

# Iteramos  sobre cada columna
for i in range(num_columnas):
    suma_columna = 0

    for j in range(num_filas):
        suma_columna += X[j][i]
        
#media por coluna    
    media_columna = suma_columna / num_filas    
    medias.append(media_columna)

#desviacion estandar
desviaciones_estandar = [math.sqrt(sum((X[j][i] - medias[i]) ** 2 for j in range(num_filas)) / num_filas) for i in range(num_columnas)]

# formula para estandarizar X
X_estandarizado = [[(X[j][i] - medias[i]) / desviaciones_estandar[i] for i in range(num_columnas)] for j in range(num_filas)]

print("\nestandarizado de x entre -1 y 1, mostrar los 15 primeros")
for i in range(15):
    print(X_estandarizado[i])
#//////////////////////////////////////////////////////////////////////////////////////



#//////////////--------ONE_HOT---------////////////////////////////////////////
# segun data set Iris
especies = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
#one_hot acortado
un_solo_bit = [[1 if i == especies[etiqueta] else 0 for i in range(len(especies))] for etiqueta in y]

"""
print("one-hot:")
for i in range(30):
    print(un_solo_bit[i])
"""    
#//////////////////////////////////////////////////////////////////////////////    




#//////////////--------MATRIZ DE PESOS, RELU Y SOFTMAX---------///////////////////
"""
El data set de iris contiene cuatro entradas.
Length of the sepal
Width of the sepal
Length of the petal
Width of the petal

primeras neuronas para la entrada random 4
"""
#Hallamos los pesos por cada capa
#(primera capa) capa de entrada, capa oculta (4 neuronas con 4 variables)
cap_escondida = [[random.uniform(-1, 1) for _ in range(4)] for _ in range(4)]
#(segunda capa) capa salida (1 salida 3 variables de entrada)
cap_salida = [[random.uniform(-1, 1) for _ in range(3)] for _ in range(3)]

print("''''''''''''''''''''''''''''''''''''''''''''''")
print(cap_escondida)
print("''''''''''''''''''''''''''''''''''''''''''''''")
print(cap_salida)
print("''''''''''''''''''''''''''''''''''''''''''''''")
#activación ReLU
def relu(x):
    return max(0, x)

#Softmax
def softmax(x):
    exp_values = [math.exp(i - max(x)) for i in x]
    sum_exp_values = sum(exp_values)
    return [i / sum_exp_values for i in exp_values]
#//////////////////////////////////////////////////////////////////////////////




tasa_aprendizaje = 0.01
epocas = 100
# Entrenamiento de la red neuronal
for epoca in range(epocas):
    error_promedio = 0

    for i in range(len(X_estandarizado)):
        # Propagación hacia adelante
        entrada = X_estandarizado[i]

        # Capas
        capa_oculta_salida = [relu(sum([entrada[j] * cap_escondida[k][j] for j in range(4)])) for k in range(4)]
        capa_salida_salida = softmax([sum([capa_oculta_salida[j] * cap_salida[k][j] for j in range(3)]) for k in range(3)])

        # Error cuadrático
        error = sum([(un_solo_bit[i][k] - capa_salida_salida[k]) ** 2 for k in range(3)]) / 2
        error_promedio += error

        # Backpropagation (o propagación hacia atrás de los errores)
        
        for k in range(3):
            for j in range(3):
                cap_salida[k][j] += tasa_aprendizaje * (un_solo_bit[i][k] - capa_salida_salida[k]) * capa_oculta_salida[j]
                
        for k in range(3):
            for j in range(4):
                sumatoria_oculta = sum([(un_solo_bit[i][l] - capa_salida_salida[l]) * cap_salida[l][k] for l in range(3)])
                cap_escondida[k][j] += tasa_aprendizaje * sumatoria_oculta * entrada[j]

       
        
    error_promedio /= len(X_estandarizado)
    print(f"Epoca {epoca + 1}, Error Promedio: {error_promedio}")

#print("\npredicción final: ",capa_salida_salida, " Valor esperado: ",un_solo_bit[149])

# valor de pesos finales
print("\nPesos de la capa oculta")
print(cap_escondida)

print("\nPesos de la capa de salida")
print(cap_salida)