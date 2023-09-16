import os
import pandas as pd
import audiofile
import opensmile
import glob
import numpy as np
import parselmouth 
from parselmouth.praat import call
from sklearn.decomposition import PCA
import featureExtraction



# Fichero que contiene extracción de datos de la base de datos SVD

# Crea un OpenSmile dataset de tipo 1 ComParE_2016 o 2 eGEMAPSv02, con sexo y patológicos
# smileConfiguration:  1 ComParE_2016 o 2 eGEMAPSv02
# filesPath: path donde se encuentran los ficheros a recorrer
# metadataPath: path donde están los metadatos
# fileSufix: sufijo de los ficheros que se van a recorrer en 01-phrase.wav, sería "phrase"
# sexConfiguration: 'm', 'w' or 'all' para todos los sexos
def _opensmileExtractionFeature(smileConfiguration:int, filesPath:str, metadataPath:str, fileSufix:str, sexConfiguration:str, minAge:int, maxAge:int, fromSec:int, toSec:int):
    if smileConfiguration == 1:
        smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.ComParE_2016,
        feature_level=opensmile.FeatureLevel.Functionals,
    )

    if smileConfiguration == 2:
      smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.Functionals,
    )

    df = pd.DataFrame()
    meta = pd.read_excel(metadataPath, sheet_name='AVFAD')
    ids = meta['File ID'].tolist()
    healthy = meta['CMVD-I Dimension 1 (numeric system)'].tolist()
    sex = meta['Sex'].tolist()
    age = meta['Age'].tolist()
    subject_unique = []

    for idx in ids:
            file = os.path.join(filesPath, str(idx)+"/"+str(idx)+fileSufix+".wav")
            print(filesPath, str(idx)+"/"+str(idx)+fileSufix+".wav")
            if(os.path.isfile(file)):
                placeId = ids.index(idx)
                if sexConfiguration !='all':
                    if sex[placeId]== sexConfiguration:
                        if featureExtraction._ageFilter(age[placeId],minAge,maxAge):
                            signal, sampling_rate = audiofile.read(file,offset = fromSec, duration=toSec,always_2d=True)
                            result_df = smile.process_signal(signal,sampling_rate)
                            result_df['age'] = age[placeId]
                            result_df['healthy'] = 0 if healthy[placeId]=="0" else 1
                            df = pd.concat([df, pd.DataFrame(result_df)], axis=0)
                else:
                    if featureExtraction._ageFilter(age[placeId],minAge,maxAge):
                        signal, sampling_rate = audiofile.read(file,duration=1,always_2d=True)
                        result_df = smile.process_signal(signal,sampling_rate)
                        result_df['age'] = age[placeId]
                        result_df['sex'] = 0 if sex[placeId]=="F" else 1
                        result_df['healthy'] = 0 if healthy[placeId]==0 else 1
                        df = pd.concat([df, pd.DataFrame(result_df)], axis=0)
    print(subject_unique)
    name = "Data_"+fileSufix+"-"+sexConfiguration+"-"+str(minAge)+"-"+str(maxAge)
    #print("DATASET: "+name+" has "+ df.loc[:, 'healthy'].tolist().size()+" records")
    #print("DATASET: "+name+" has "+ df.loc[:, 'healthy'].tolist().count(0)+" healthy and "+df.loc[:, 'healthy'].tolist().count(1)+" pathological")
    #if sexConfiguration =='all':
    #    print("DATASET: "+name+" has "+ df.loc[:, 'sex'].tolist().count(0)+" women and "+df.loc[:, 'sex'].tolist().count(1)+" men")
    features_dataframe_no_nan = df.dropna(axis=0 , how='any')
    features_dataframe_no_nan.to_csv(name+".csv", index=False)





