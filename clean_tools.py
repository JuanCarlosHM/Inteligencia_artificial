# Proyecto de inteligencia artificial: 
# "Sistema de recomendación"
# ====================================
# Elaborado por: 
# @Ulisies 
# @Juan Carlos Herrera Mercado 
# @
# @ 
#=====================================
# Laas referencias son citadas al final
# de este archivo. 
#


# import statments
import pickle 
import numpy as np                                                      
import pandas as pd                                                    
import matplotlib.pyplot as plt   
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer


# Leemos la información del data set de pinturas 
# El data set ya fue tratado previamente. 
# El data set contine registro de aprox. 75 mil pinturas 
paintings_ds = pd.read_csv("data/clear_data_set.csv")

# ========================================================================================================
#                                           Limpieza de datos 
# ========================================================================================================
# Limpiamos el dat set para eliminar los registros  con campos vacios 
paintings_ds = paintings_ds.dropna(how='any')

# Limpiamos los datos de caracteres no estndar. 
paintings_ds['genre'] = paintings_ds['genre'].replace({"-": " "}, regex=True)
paintings_ds['title'] = paintings_ds['title'].replace({"-": " "}, regex=True)
paintings_ds['artist'] = paintings_ds['artist'].replace({"-": " "}, regex=True)
paintings_ds['styleTipo'] = paintings_ds['styleTipo'].replace({"-": " "}, regex=True)

# Verificamos el tipo de los campos del data set
print(paintings_ds.dtypes)




# ========================================================================================================
#                                          Manejo de los datos 
# ========================================================================================================

# Combinamos solo algunos elementos del data set
# comb_frame = paintings_ds.artist.str.cat(" "+paintings_ds.date.astype(str).str.cat(" "+paintings_ds.genre))

#combinación de todos los campos del dataset en un solo vector 
comb_frame_all_fields = paintings_ds.artist.str.cat(" "+paintings_ds.date.astype(str).str.cat(" "+paintings_ds.genre.astype(str).str.cat(" "+paintings_ds.styleTipo.astype(str).str.cat(" "+paintings_ds.title))))

# Verificamos la salida obtenida
#print(comb_frame)
#print(comb_frame_all_fields)

# Eliminamos todos los caracteres que no sean letras o numeros
# comb_frame = comb_frame.replace({"[^A-Za-z0-9 ]+": ""}, regex=True)
comb_frame_all_fields = comb_frame_all_fields.replace({"[^A-Za-z0-9 ]+" : ""}, regex=True)


vectorizer = TfidfVectorizer(stop_words='english')
#vectorizer = TfidfVectorizer()


# X = vectorizer.fit_transform(comb_frame)
Y = vectorizer.fit_transform(comb_frame_all_fields)


# Despues de vectorizar los datos, creamos una estructura que 
# contenga la suma de errores de los cuadrados de los vaores obtenidos (Sum-Of-Square-Errors) 
# al iterar el algoritmo k-means 80 veces. Con el fin de poder obtener el numero de 
# K idelaes sin sobre entrenar el modelo. 

# (Sum-Of-Square-Errors)
sse = {}


# este bloque es tremendamente costoso en tiempo, ejecutar si se tiene dida acerca de los clusters

"""
for k in range(1, 50):
    kmeans = KMeans(n_clusters=k, init='k-means++', max_iter=10).fit(Y)
    comb_frame["clusters"] = kmeans.labels_
    sse[k] = kmeans.inertia_

plt.plot(list(sse.keys()), list(sse.values()))
plt.xlabel("Número de clusters")
plt.ylabel("Sum-Of-Square-Errors")

# Grafica oftenida para determinar mediante "elbow-test" el número ideal de k
plt.savefig('elbow_method.png')
"""

# La k determinada por el método del codo 
true_k = 45

# Construimos el modelo a partir de 15 difentes inicialiaciones de centroides y una iteración máxima de 500
model = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=15)
model.fit(Y)


# ========================================================================================================
#                                          Manejo de Clusters 
# ========================================================================================================


# 5. Preview clusters, test your model for demo and save your model for further use.

# Preview top 15 words in each cluster, and accordingly different clusters can be assigned 

# Create a hashmap, mapping each cluster to a given category.

print("Top terms per cluster:")
order_centroids = model.cluster_centers_.argsort()[:, ::-1]
terms = vectorizer.get_feature_names()
for i in range(true_k):
    print("Cluster %d:" % i),
    for ind in order_centroids[i, :15]:
        print(' %s' % terms[ind]),
    print

# Save machine learning model
filename = 'finalized_model.sav'
pickle.dump(model, open(filename, 'wb'))








