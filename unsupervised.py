# -*- coding: utf-8 -*-

''' RIASC Automated Decision Support Software (RADSSo) 1.0 generates the best supervised/unsupervised model,
    in an automated process, based on some input features and one target feature, to solve a multi-CASH problem.

    Copyright (C) 2018  by RIASC Universidad de Leon (Ángel Luis Muñoz Castañeda, Mario Fernández Rodríguez, Noemí De Castro García y Miguel Carriegos Vieira)
    This file is part of RIASC Automated Decision Support Software (RADSSo) 1.0

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
	
	You can find more information about the project at https://github.com/amunc/RADSSo1.0'''
    
import os
import numpy as np
from sklearn.externals import joblib
from sklearn.cluster import KMeans

import auxiliary_functions as auxf


def compare_major_classes(diccionario_clases):
    '''Compare the classes to get the major one'''
    mayoritaria= ''
    for key in diccionario_clases:
        if mayoritaria == '':
            mayoritaria = key
        elif(diccionario_clases[key] > diccionario_clases[mayoritaria]):
            mayoritaria = key
    return key
            

def create_customized_kmeans(train_data,features,target,num_clusters,numero_inicializaciones):
#la funcion mi_kmeans permite crear el modelo KMeans usando la libreria sklearn
#recibe como parametros: train_data(los datos de entrenamiento), features(las features relevantes), target(feature objetivo), num_clusters(numero de clusters en los que debe agrupar los datos), numero_inicializaciones( Numero de veces que el algoritmo k-means se ejecuta con difierentes inicializaciones de los centroid, tomando el mejor de ellos)
#devuelve kmeans(el modelo Kmeans entrenado)  
    X=train_data[features]
    y=train_data[target]
    if(num_clusters == '' and numero_inicializaciones == ''):
        kmeans = KMeans(init='k-means++')        
    else:
        kmeans = KMeans(n_clusters=num_clusters,init='k-means++',n_init = numero_inicializaciones)        
    kmeans=kmeans.fit(X,y)
    return(kmeans)


def get_dictionary_of_reasignation_of_labels(modelo,dataset,target):
#la funcion mi_spectral_clustering permite crear el modelo Spectral clustering, y entrenarlo, usando la libreria sklearn
#recibe como parametros: train_data(los datos de entrenamiento), features(las features relevantes), target(feature objetivo), num_clusters(numero de clusters en los que debe agrupar los datos), numero_inicializaciones( Numero de veces que el algoritmo k-means se ejecuta con difierentes inicializaciones de los centroid, tomando el mejor de ellos)
#devuelve kmeans(el modelo Kmeans entrenado)   
    diccionario_plantilla = {}    
    clusters = set(list(modelo.labels_))
    relacion_cluster_clase = {}
    clases_originales = list(set(dataset[target]))
    diccionario_clases_originales_contador = {}
    for clase_original in clases_originales:
        diccionario_clases_originales_contador[clase_original] = 0
        
    for cluster in clusters:
        diccionario_plantilla[cluster] = diccionario_clases_originales_contador.copy() #plantilla con los tipos de observaciones que hay de cada target dentro del nuevo cluster
        
        indices_cluster = list(np.where(modelo.labels_ == cluster)[0])
        
        diccionario_contador={}
        relacion_cluster_clase[cluster] = []
        acum = 0
        for indice in indices_cluster:        
            clase = dataset.iloc[int(indice)][target]

            if(clase not in diccionario_contador):
                diccionario_contador[clase] = 1
                acum+=1
            else:
                valores = diccionario_contador[clase]
                valores+=1
                acum+=1
                diccionario_contador[clase] = valores        
                            
        for clase_original in diccionario_contador:
            diccionario_plantilla[cluster][clase_original] = diccionario_contador[clase_original]
    
    targets_originales_ordenados = []
    for indice in range(len(clases_originales)):
        for cluster in diccionario_plantilla:
            mayoritario = ''
            numero_mayoritario = 0
            for target_original in diccionario_plantilla[cluster]:
                if(target_original not in targets_originales_ordenados):
                    if(mayoritario == ''):
                        mayoritario = target_original
                        numero_mayoritario = diccionario_plantilla[cluster][mayoritario]
                    else:
                        numero_candidato = diccionario_plantilla[cluster][target_original]
                        if(numero_candidato > numero_mayoritario):
                            numero_mayoritario = numero_candidato
                            mayoritario = target_original
            if(mayoritario != ''):
                targets_originales_ordenados.append(mayoritario)
            
    clases_originales = targets_originales_ordenados

    asociacion_cluster_target={}        
    for clase in clases_originales:
        elementos_cluster_clase = {}
        for cluster_actual in clusters:
            elementos_cluster_clase[cluster_actual] = diccionario_plantilla[cluster_actual][clase]

        mayoritaria = ''
        for cluster in elementos_cluster_clase:
            if(mayoritaria == ''):
                mayoritaria = cluster
            else: #comprobamos
                actual = elementos_cluster_clase[mayoritaria]
                candidata = elementos_cluster_clase[cluster]
                if(candidata > actual):
                    mayoritaria = cluster

        if(mayoritaria != ''):
            clusters.remove(mayoritaria)#eliminamos el cluster de la lista completa de clusters
            asociacion_cluster_target[mayoritaria] = clase
   
    diccionario_plantilla_recodificado = {}
    for cluster in asociacion_cluster_target:
        reco = asociacion_cluster_target[cluster]
        diccionario_plantilla_recodificado[reco] = diccionario_plantilla[cluster]
    
    return asociacion_cluster_target

def recodify_list_predicted_targets(lista_predicciones, diccionario_reco):
    '''Recodification of predicted targets using dictionary'''    
    predicciones_reco = []
    for elemento in lista_predicciones:
        reco = diccionario_reco[elemento]
        predicciones_reco.append(reco)
    return predicciones_reco


def get_accuracy(lista_targets_datos_catalogados,targets_predichos_recodificados):
    #la funcion permite obtener el porcentaje de acierto del entrenamiento del modelo *kmeans
    #recibe lista_targets_data_set(el conjunto de datos completo con la feature objetivo), features (features relevantes), target (la feature objetivo)
    #devuelve el porcentaje de acierto
    accuracy=auxf.compare_lists(targets_predichos_recodificados,lista_targets_datos_catalogados)/float(len(lista_targets_datos_catalogados))
    return(accuracy)
    
def save_model_to_disk(modelo,ruta_directorio,nombre_fichero):
    '''
    Permite generar un fichero .pkl con el modelo de aprendizaje generado usando scikit-learn

    Parameters:
    :param sklearn-model model: modelo de aprendizaje obtenido mediante el uso de scikit-learn    
    :param str ruta_directorio: ruta completa al directorio donde se quiere guardar el modelo
    :param str nombre_fichero: nombre del fichero que almacenará el modelo (debe incluir la extension)    
    
    :return: None
    '''
    
    ruta_destino = os.path.join(ruta_directorio,nombre_fichero)
    joblib.dump(modelo, ruta_destino)
    return ruta_destino
    
def initialize_model(model_name,params_array,diccionario_modelos_no_supervisado):
    '''
    The function allows to create and train a model with the specified name

    Parameters:
    :param str model_name: name of the model to be trained
    :param pandas-dataframe train_data: Data to train de model. It includes the target column
    :param list features: List with the relevant features to rain the model
    :param str target: Target feature
    :param list params_array: Array with the specific parameters of the model to be trained
    
    :return: specified model trained
    :rtype: sklearn-model    
    
    '''
    
    modelo_inicializado = ''
    if(model_name == diccionario_modelos_no_supervisado[1]):#Kmeans
        modelo_inicializado = KMeans(n_clusters=params_array[0],n_init=params_array[1])
            
    return modelo_inicializado

def create_trained_model(model_name,train_data,features,target,params_array,diccionario_modelos_no_supervisado):
    '''
    The function allows to create and train a model with the specified name

    Parameters:
    :param str model_name: name of the model to be trained
    :param pandas-dataframe train_data: Data to train de model. It includes the target column
    :param list features: List with the relevant features to rain the model
    :param str target: Target feature
    :param list params_array: Array with the specific parameters of the model to be trained
    
    :return: specified model trained
    :rtype: sklearn-model
    '''
    
    modelo_creado = ''
    if(model_name == diccionario_modelos_no_supervisado[1]):#Kmeans
        modelo_creado = create_customized_kmeans(train_data,features,target,params_array[0],params_array[1])
        
    return modelo_creado
