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

# import statements
import pandas as pd
import pickle
import random

from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer


# Abrimos el modelo creado con anterioridad y lo cargamos en memoria para trabajar con el 
with open('finalized_model.sav', 'rb') as fid:
    model = pickle.load(fid)

# Tomamos el data set original 
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

# Estandarizamos la información en un solo campo
comb_frame_all_fields = paintings_ds.artist.str.cat(" "+paintings_ds.date.astype(str).str.cat(" "+paintings_ds.genre.astype(str).str.cat(" "+paintings_ds.styleTipo.astype(str).str.cat(" "+paintings_ds.title))))

# Eliminamos caracteres que no son números y letras 
comb_frame_all_fields = comb_frame_all_fields.replace({"[^A-Za-z0-9 ]+": ""}, regex=True)

# Vectorizamos la información 
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(comb_frame_all_fields)
    

def cluster_predict(str_input):
    Y = vectorizer.transform(list(str_input))
    prediction = model.predict(Y)
    return prediction


# No sense 
#paintings_ds = pd.read_csv("data/clear_data_set.csv")

paintings_ds['vectorOfData'] = paintings_ds.artist.str.cat(" "+paintings_ds.date.astype(str).str.cat(" "+paintings_ds.genre.astype(str).str.cat(" "+paintings_ds.styleTipo.astype(str).str.cat(" "+paintings_ds.title))))

paintings_ds['ClusterPrediction'] = ""

paintings_ds['ClusterPrediction'] = paintings_ds.apply(lambda x: cluster_predict(paintings_ds['vectorOfData']), axis=0)



def recommend_util(str_input):
    
    # Predict category of input string category
    temp_ds = paintings_ds.loc[paintings_ds['title'] == str_input]

    temp_ds['InputString'] = temp_ds.artist.str.cat(" "+temp_ds.date.astype(str).str.cat(" "+temp_ds.genre.astype(str).str.cat(" "+temp_ds.styleTipo.astype(str).str.cat(" "+temp_ds.title))))
    
    str_input = list(temp_ds['InputString'])
    
    prediction_inp = cluster_predict(str_input)
    print("here - flag:")
    print(prediction_inp)
    
    prediction_inp = int(prediction_inp)

    # Based on the above prediction 10 random courses are recommended from the whole data-frame
    # Recommendation Logic is kept super-simple for current implementation.    
    temp_ds = paintings_ds.loc[paintings_ds['ClusterPrediction'] == prediction_inp]

    temp_ds = temp_ds.sample(10)
    
    return list(temp_ds['title'])



# ========================================================================================================
#                                   Método que pidío frontend 
# ========================================================================================================
def recommendation_randomly_list(numbers): 
    random_subset = paintings_ds.sample(n=numbers)
    print(random_subset.head())
    return random_subset





# ========================================================================================================
# aqui agrega los datos de las pinturas, por ejempo, yo ya probe con estos 2 cuadros, solo funciona con el titulo!
# la respuesta serán 10 titulos tomados de manera random del cluster en el que se encuetre el titulo de la obra
# obvio se deben tomar los titulos de nuestro data set, jala hermoso :´) 
# ========================================================================================================

if __name__ == '__main__':
    queries = ['Uriel', 'Vir Heroicus Sublimis']

    recommendation_randomly_list(10)

    for query in queries:
        res = recommend_util(query)
        print(res)