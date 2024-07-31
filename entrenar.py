import sys
import os
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras import optimizers
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dropout,Flatten,Dense,Activation
from tensorflow.python.keras.layers import Convolution2D,MaxPooling2D
from tensorflow.python.keras import backend as K

#ImageDataGenerator -> Ayuda a preprocesar las imagenes
#optimizers -> optimizador para entrenar el algoritmo
#Sequential -> Libreria que permite hacer redes neuronales secuenciales
#Dropout,Flatten,Dense,Activation
#Convolution2d -> Capas de las convoluciones
#Backend -> Si hay una sesion de keras corriendo lo finaliza para empezar de 0

K.clear_session()
data_entrenamiento = './data/entrenamiento'
data_validation = './data/validacion'

##parametros de la red
epocas = 20 #numero de iteracion de la red
altura,longitud = 100,100 #tamaño para procesar las imagenes
batch_size = 32 #Numero de imagenes en cada uno de los pasos
pasos=1000 #Numero de veces en que se procesa la informacion
pasos_validacion = 200 #Cantidad de pasos para el set de datos de validacion
filtrosConv1=32
filtrosConv2=64
tamano_filtro1 = (3,3)
tamano_filtro2 = (2,2)
tamano_pool=(2,2) #tamaño del filtro para el max pooling
clases = 3 #la cantidad de imagenes, en este caso tenemos que definir la cantidad
# de señales que vamos a identificar
lr = 0.005 #tamaño del ajuste para acercarse a una solucion optima

##pre procesamiento de imagenes

entrenamiento_datagen = ImageDataGenerator(
    rescale = 1./255,
    shear_range= 0.3,
    zoom_range = 0.3,
    horizontal_flip = True
)

validacion_datagen = ImageDataGenerator(
    rescale = 1./255
)

imagen_entrenamiento = entrenamiento_datagen.flow_from_directory(
    data_entrenamiento,
    target_size = (altura,longitud),
    batch_size = batch_size,
    class_mode = 'categorical'
)

imagen_validacion = validacion_datagen.flow_from_directory(
    data_validation,
    target_size = (altura,longitud),
    batch_size = batch_size,
    class_mode = 'categorical'
)


#CNN
cnn = Sequential()
cnn.add(Convolution2D(filtrosConv1, tamano_filtro1, padding='same',input_shape=(altura,longitud,3),activation='relu'))
cnn.add(MaxPooling2D(pool_size=tamano_pool))

cnn.add(Convolution2D(filtrosConv2, tamano_filtro2, padding='same',activation='relu'))
cnn.add(MaxPooling2D(pool_size=tamano_pool))

#Aplanar la imagen
cnn.add(Flatten())
#Cambiamos la capa a una normal, osea conectamos las 256 neuronas 
cnn.add(Dense(256,activation='relu'))
cnn.add(Dropout(0.5))
#Las clases que es la cantiad que vamos analizar,softmax da los porcentajes
#Osea nos dirá este % es tal pare, es tal alto, izquierda, derecha, etc
cnn.add(Dense(clases, activation='softmax'))
#Intentará mejorar el accuaracy osea el porcentaje de cada imagen 
cnn.compile(loss='categorical_crossentropy',optimizer=optimizers.Adam(lr=lr),metrics = ['accuracy'])

#Entrenamiento de las imagenes
cnn.fit_generator(imagen_entrenamiento,steps_per_epoch=pasos, epochs=epocas, validation_data= imagen_validacion, validation_steps=pasos_validacion)

dir = './modelo/'

if not os.path.exists(dir):
    os.mkdir(dir)

cnn.save('./modelo/modelo.h5')
cnn.save_weights('./modelo/pesos.h5')
    
    
    
    
    
    
    
    
    
    