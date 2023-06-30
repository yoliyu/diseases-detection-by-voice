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
def _opensmileExtractionFeature(smileConfiguration:int, filesPath:str, metadataPath:str, fileSufix:str, sexConfiguration:str, minAge:int, maxAge:int):
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
    path = filesPath
    meta = pd.read_excel(metadataPath, sheet_name='SVD')
    ids = meta['ID'].tolist()
    healthy = meta['Healthy'].tolist()
    sex = meta['Sex'].tolist()
    age = meta['Age'].tolist()
    subject = meta['Subject'].tolist()
    subject_unique = []

    for idx in ids:
            file = os.path.join(filesPath, str(idx)+"-"+fileSufix+".wav")
            if(os.path.isfile(file)):
                placeId = ids.index(int(idx))
                if  subject[placeId] in subject_unique != True:
                    if sexConfiguration !='all':
                        if sex[placeId]== sexConfiguration:
                            if featureExtraction._ageFilter(age[placeId],minAge,maxAge):
                                signal, sampling_rate = audiofile.read(file,duration=1,always_2d=True)
                                result_df = smile.process_signal(signal,sampling_rate)
                                result_df['age'] = age[placeId]
                                result_df['healthy'] = 0 if healthy[placeId]=="n" else 1
                                df = pd.concat([df, pd.DataFrame(result_df)], axis=0)
                    else:
                        if featureExtraction._ageFilter(age[placeId],minAge,maxAge):
                            signal, sampling_rate = audiofile.read(file,duration=1,always_2d=True)
                            result_df = smile.process_signal(signal,sampling_rate)
                            result_df['age'] = age[placeId]
                            result_df['sex'] = 0 if sex[placeId]=="m" else 1
                            result_df['healthy'] = 0 if healthy[placeId]=="n" else 1
                            df = pd.concat([df, pd.DataFrame(result_df)], axis=0)
                subject_unique.append(subject[placeId])
    print(subject_unique)
    name = "Data_"+fileSufix+"-"+sexConfiguration+"-"+str(minAge)+"-"+str(maxAge)
    #print("DATASET: "+name+" has "+ df.loc[:, 'healthy'].tolist().size()+" records")
    #print("DATASET: "+name+" has "+ df.loc[:, 'healthy'].tolist().count(0)+" healthy and "+df.loc[:, 'healthy'].tolist().count(1)+" pathological")
    #if sexConfiguration =='all':
    #    print("DATASET: "+name+" has "+ df.loc[:, 'sex'].tolist().count(0)+" women and "+df.loc[:, 'sex'].tolist().count(1)+" men")
    features_dataframe_no_nan = df.dropna(axis=0 , how='any')
    features_dataframe_no_nan.to_csv(name+".csv", index=False)


# Crea un dataset con duration, meanF0, stdevF0, meanHNR, stdevHNR, localJitter, localabsoluteJitter, 
# rapJitter, ppq5Jitter, ddpJitter, localShimmer, localdbShimmer, apq3Shimmer, aqpq5Shimmer, apq11Shimmer, ddaShimmer
# f1_mean, f2_mean, f3_mean, f4_mean, jitterPCA, shimmerPCA
# start 0.0
# end 2.0
# timeStep 0.0
# unit "Hertz"
# filesPath: path donde se encuentran los ficheros a recorrer
# metadataPath: path donde están los metadatos
# fileSufix: sufijo de los ficheros que se van a recorrer en 01-phrase.wav, sería "phrase"
def _regularExtractionFeature(start: int, end:int, timeStep:int, unit:str, filesPath:str, metadataPath:str, fileSufix:str):
    # create lists to put the results
    file_list = []
    duration_list = []
    mean_F0_list = []
    sd_F0_list = []
    meanHNR_list = []
    stdevHNR_list = []
    localJitter_list = []
    localabsoluteJitter_list = []
    rapJitter_list = []
    ppq5Jitter_list = []
    ddpJitter_list = []
    localShimmer_list = []
    localdbShimmer_list = []
    apq3Shimmer_list = []
    aqpq5Shimmer_list = []
    apq11Shimmer_list = []
    ddaShimmer_list = []
    f1_mean_list = []
    f2_mean_list = []
    f3_mean_list = []
    f4_mean_list = []

    sex_list = []
    healthy_list = []
    age_list = []

    meta = pd.read_excel(metadataPath, sheet_name='SVD')
    ids = meta['ID'].tolist()
    healthy = meta['Healthy'].tolist()
    sex = meta['Sex'].tolist()
    age = meta['Age'].tolist()
    for wave_file in glob.glob(filesPath+"*.wav"):
        x = wave_file.replace(filesPath, "")
        id = x.replace("-"+fileSufix+".wav", "")
        id = id.replace("\\", "")
        print(id)
        placeId = ids.index(int(id))
        age_list.append(age[placeId])
        sex_list.append(0 if sex[placeId]=="m" else 1)
        healthy_list.append(0 if healthy[placeId]=="n" else 1)
        f0min = 60 if sex[placeId]=="m" else 100
        f0max = 300 if sex[placeId]=="m" else 500

        sound = parselmouth.Sound(wave_file)
        (duration, meanF0, stdevF0, meanHNR, stdevHNR, localJitter, localabsoluteJitter, rapJitter, ppq5Jitter, ddpJitter, 
        localShimmer, localdbShimmer, apq3Shimmer, aqpq5Shimmer, apq11Shimmer, ddaShimmer) = featureExtraction._measurePitch(sound, f0min, f0max, unit, start, end, timeStep)
        (f1_mean, f2_mean, f3_mean, f4_mean) = featureExtraction._measureFormants2(sound, wave_file,f0min,f0max,start,end,unit)
        
        file_list.append(id) # make an ID list
        duration_list.append(duration) # make duration list
        mean_F0_list.append(meanF0) # make a mean F0 list
        sd_F0_list.append(stdevF0) # make a sd F0 list
        meanHNR_list.append(meanHNR) #add HNR data
        stdevHNR_list.append(stdevHNR) #add HNR data
        
        # add raw jitter and shimmer measures
        localJitter_list.append(localJitter)
        localabsoluteJitter_list.append(localabsoluteJitter)
        rapJitter_list.append(rapJitter)
        ppq5Jitter_list.append(ppq5Jitter)
        ddpJitter_list.append(ddpJitter)
        localShimmer_list.append(localShimmer)
        localdbShimmer_list.append(localdbShimmer)
        apq3Shimmer_list.append(apq3Shimmer)
        aqpq5Shimmer_list.append(aqpq5Shimmer)
        apq11Shimmer_list.append(apq11Shimmer)
        ddaShimmer_list.append(ddaShimmer)
        
        # add the formant data
        f1_mean_list.append(f1_mean)
        f2_mean_list.append(f2_mean)
        f3_mean_list.append(f3_mean)
        f4_mean_list.append(f4_mean)
    


    # Add the data to Pandas
    features_dataframe_nan = pd.DataFrame(np.column_stack([sex_list, healthy_list, age_list,duration_list, mean_F0_list, sd_F0_list, meanHNR_list, stdevHNR_list, 
                                    localJitter_list, localabsoluteJitter_list, rapJitter_list, 
                                    ppq5Jitter_list, ddpJitter_list, localShimmer_list, 
                                    localdbShimmer_list, apq3Shimmer_list, aqpq5Shimmer_list, 
                                    apq11Shimmer_list, ddaShimmer_list, f1_mean_list, 
                                    f2_mean_list, f3_mean_list, f4_mean_list]),
                                    columns=['sex', 'healthy', 'age','duration','meanF0Hz', 'stdevF0Hz', 'meanHNR', 'stdevHNR', 
                                                'localJitter', 'localabsoluteJitter', 'rapJitter', 
                                                'ppq5Jitter', 'ddpJitter', 'localShimmer', 
                                                'localdbShimmer', 'apq3Shimmer', 'apq5Shimmer', 
                                                'apq11Shimmer', 'ddaShimmer', 'f1_mean', 'f2_mean', 
                                                'f3_mean', 'f4_mean'])

    features_dataframe = features_dataframe_nan.dropna(axis=0 , how='any')
    pca_dataframe = featureExtraction._runPCA(features_dataframe) # Run jitter and shimmer PCA
    features_dataframe = pd.concat([features_dataframe,pca_dataframe], axis=1) # Add PCA data
    features_dataframe_no_nan = features_dataframe.dropna(axis=0 , how='any')
    features_dataframe_no_nan.to_csv("X_"+fileSufix+".csv", index=False)






