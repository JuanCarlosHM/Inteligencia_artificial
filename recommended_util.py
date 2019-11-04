# ========================================================================================================
#                                          PENDIENTE DE DOCUMENTAR  
# ========================================================================================================

# import statements
import pandas as pd
import pickle
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer



with open('finalized_model.sav', 'rb') as fid:
    model = pickle.load(fid)


paintings_ds = pd.read_csv("data/clear_data_set.csv")

paintings_ds = paintings_ds.dropna(how='any')

# Limpiamos los datos de caracteres no estndar. 
paintings_ds['genre'] = paintings_ds['genre'].replace({"-": " "}, regex=True)
paintings_ds['title'] = paintings_ds['title'].replace({"-": " "}, regex=True)
paintings_ds['artist'] = paintings_ds['artist'].replace({"-": " "}, regex=True)
paintings_ds['styleTipo'] = paintings_ds['styleTipo'].replace({"-": " "}, regex=True)

comb_frame_all_fields = paintings_ds.artist.str.cat(" "+paintings_ds.date.astype(str).str.cat(" "+paintings_ds.genre.astype(str).str.cat(" "+paintings_ds.styleTipo.astype(str).str.cat(" "+paintings_ds.title))))

comb_frame_all_fields = comb_frame_all_fields.replace({"[^A-Za-z0-9 ]+": ""}, regex=True)

vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(comb_frame_all_fields)
    



paintings_ds = pd.read_csv("data/clear_data_set.csv")



paintings_ds['vectorOfData'] = paintings_ds.artist.str.cat(" "+paintings_ds.date.astype(str).str.cat(" "+paintings_ds.genre.astype(str).str.cat(" "+paintings_ds.styleTipo.astype(str).str.cat(" "+paintings_ds.title))))

paintings_ds['ClusterPrediction'] = ""


def cluster_predict(str_input):
    Y = vectorizer.transform(list(str_input))
    prediction = model.predict(Y)
    return prediction

paintings_ds['ClusterPrediction']=paintings_ds.apply(lambda x: cluster_predict(paintings_ds['vectorOfData']), axis=0)


def recommend_util(str_input):
    
    # Predict category of input string category
    temp_ds = paintings_ds.loc[paintings_ds['title'] == str_input]

    temp_ds['InputString'] = temp_ds.artist.str.cat(" "+temp_ds.date.astype(str).str.cat(" "+temp_ds.genre.astype(str).str.cat(" "+temp_ds.styleTipo.astype(str).str.cat(" "+temp_ds.title))))
    str_input = list(temp_ds['InputString'])
    
    prediction_inp = cluster_predict(str_input)
    prediction_inp = int(prediction_inp)
    
    temp_ds = paintings_ds.loc[paintings_ds['ClusterPrediction'] == prediction_inp]
    temp_ds = temp_ds.sample(10)
    
    return list(temp_ds['title'])



# ========================================================================================================
# aqui agrega los datos de las pinturas, por ejempo, yo ya probe con estos 2 cuadros, solo funciona con el titulo!
# la respuesta serán 10 titulos tomados de manera random del cluster en el que se encuetre el titulo de la obra
# obvio se deben tomar los titulos de nuestro data set, jala hermoso :´) 
# ========================================================================================================

if __name__ == '__main__':
    queries = ['Uriel', 'Vir Heroicus Sublimis']

    for query in queries:
        res = recommend_util(query)
        print(res)