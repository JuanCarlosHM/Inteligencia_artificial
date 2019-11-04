
# import statements
import pandas as pd
import pickle
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer


# 1. load model and previous preprocessing.

# load model only once
with open('finalized_model.sav', 'rb') as fid:
    model = pickle.load(fid)

# X = vectorizer.fit_transform(paintings_ds['InputString'])
# This will give an error as incorrect number of features, i.e. if features from a different data-frame is used
# seperate code snippet for building vocabulary for trained model
paintings_ds = pd.read_csv("data/clear_data_set.csv")

paintings_ds = paintings_ds.dropna(how='any')

# Limpiamos los datos de caracteres no estndar. 
paintings_ds['genre'] = paintings_ds['genre'].replace({"-": " "}, regex=True)
paintings_ds['title'] = paintings_ds['title'].replace({"-": " "}, regex=True)
paintings_ds['artist'] = paintings_ds['artist'].replace({"-": " "}, regex=True)
paintings_ds['styleTipo'] = paintings_ds['styleTipo'].replace({"-": " "}, regex=True)

comb_frame_all_fields = paintings_ds.artist.str.cat(" "+paintings_ds.date.astype(str).str.cat(" "+paintings_ds.genre.astype(str).str.cat(" "+paintings_ds.styleTipo.astype(str).str.cat(" "+paintings_ds.title))))

comb_frame_all_fields = comb_frame_all_fields.replace({"[^A-Za-z0-9 ]+": ""}, regex=True)

# Add clustering labels to every non-retired course
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(comb_frame_all_fields)
    

# 2. Current utility variable and data frame preprocessing

# Verbose only, Not getting used in the code: creating labels for clusters manually
"""
label_dict = {
                 0: "Gaming Professionals",
                 1: "Manufacturing & Design",
                 2: "Software Development",
                 3: "Data Professionals",
                 4: "Information & Cyber Security",
                 5: "Movie Making & Animation",
                 6: "IT Ops",
                 7: "Graphic Design"
            }
"""

# load the complete data in a dataframe
paintings_ds = pd.read_csv("data/clear_data_set.csv")



# create new column in dataframe which is combination of (CourseId, CourseTitle, Description) in existing data-frame
paintings_ds['vectorOfData'] = paintings_ds.artist.str.cat(" "+paintings_ds.date.astype(str).str.cat(" "+paintings_ds.genre.astype(str).str.cat(" "+paintings_ds.styleTipo.astype(str).str.cat(" "+paintings_ds.title))))

paintings_ds['ClusterPrediction'] = ""


def cluster_predict(str_input):
    Y = vectorizer.transform(list(str_input))
    prediction = model.predict(Y)
    return prediction

# Cluster category for each live course
paintings_ds['ClusterPrediction']=paintings_ds.apply(lambda x: cluster_predict(paintings_ds['vectorOfData']), axis=0)


def recommend_util(str_input):
    
    # Predict category of input string category
    temp_ds = paintings_ds.loc[paintings_ds['title'] == str_input]

    #temp_ds['InputString'] = temp_ds.title.str.cat(" "+temp_df.CourseTitle.str.cat(" "+temp_df['Description']))
    temp_ds['InputString'] = temp_ds.artist.str.cat(" "+temp_ds.date.astype(str).str.cat(" "+temp_ds.genre.astype(str).str.cat(" "+temp_ds.styleTipo.astype(str).str.cat(" "+temp_ds.title))))
    str_input = list(temp_ds['InputString'])
    
    prediction_inp = cluster_predict(str_input)
    prediction_inp = int(prediction_inp)
    
    temp_ds = paintings_ds.loc[paintings_ds['ClusterPrediction'] == prediction_inp]
    temp_ds = temp_ds.sample(10)
    
    return list(temp_ds['title'])


if __name__ == '__main__':
    queries = ['Uriel', 'Vir Heroicus Sublimis']

    for query in queries:
        res = recommend_util(query)
        print(res)