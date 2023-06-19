#!/usr/bin/env python3
import glob
import numpy as np
import pandas as pd
import parselmouth 
import statistics


from parselmouth.praat import call
from scipy.stats.mstats import zscore
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler



# This is the function to measure source acoustics.
def measurePitch(voiceID, f0min, f0max, unit, start, end, timeStep):
    sound = parselmouth.Sound(voiceID) # read the sound
    duration = call(sound, "Get total duration") # duration
   
    #Extract Pitch Paramenters
    pitch = call(sound, "To Pitch", timeStep, f0min, f0max) #create a praat pitch object
    meanF0 = call(pitch, "Get mean", start, end, unit) # get mean pitch
    stdevF0 = call(pitch, "Get standard deviation", start ,end, unit) # get standard deviation
    minF0 = call(pitch, "Get minimum", start ,end, unit, "Parabolic") # get min
    maxF0 = call(pitch, "Get maximum", start ,end, unit, "Parabolic") # get max

    #This part measures harmonics to noise ratio
    # 0.01 60 0.1 4.5
    harmonicity = call(sound, "To Harmonicity (cc)", 0.01, f0min, 0.1, 1.0)
    meanHNR = call(harmonicity, "Get mean", start, end)
    stdevHNR = call(harmonicity, "Get standard deviation", start, end)

    #This part measures jitter & shimmer
    pointProcess = call(sound, "To PointProcess (periodic, cc)", f0min, f0max)
    
    localJitter = call(pointProcess, "Get jitter (local)", start, end, 0.0001, 0.02, 1.3)
    localabsoluteJitter = call(pointProcess, "Get jitter (local, absolute)", start, end, 0.0001, 0.02, 1.3)
    rapJitter = call(pointProcess, "Get jitter (rap)", start, end, 0.0001, 0.02, 1.3)
    ppq5Jitter = call(pointProcess, "Get jitter (ppq5)", start, end, 0.0001, 0.02, 1.3)
    ddpJitter = call(pointProcess, "Get jitter (ddp)", start, end, 0.0001, 0.02, 1.3)

    localShimmer =  call([sound, pointProcess], "Get shimmer (local)", start, end, 0.0001, 0.02, 1.3, 1.6)
    localdbShimmer = call([sound, pointProcess], "Get shimmer (local_dB)", start, end, 0.0001, 0.02, 1.3, 1.6)
    apq3Shimmer = call([sound, pointProcess], "Get shimmer (apq3)", start, end, 0.0001, 0.02, 1.3, 1.6)
    aqpq5Shimmer = call([sound, pointProcess], "Get shimmer (apq5)", start, end, 0.0001, 0.02, 1.3, 1.6)
    apq11Shimmer =  call([sound, pointProcess], "Get shimmer (apq11)", start, end, 0.0001, 0.02, 1.3, 1.6)
    ddaShimmer = call([sound, pointProcess], "Get shimmer (dda)", start, end, 0.0001, 0.02, 1.3, 1.6)
    
    return duration, meanF0, stdevF0, meanHNR, stdevHNR, localJitter, localabsoluteJitter, rapJitter, ppq5Jitter, ddpJitter, localShimmer, localdbShimmer, apq3Shimmer, aqpq5Shimmer, apq11Shimmer, ddaShimmer


# This function measures formants using Formant Position formula
def measureFormants1(sound, wave_file,f0min,f0max,unit):
    sound = parselmouth.Sound(sound) # read the sound
    pitch = call(sound, "To Pitch (cc)", 0, f0min, 15, 'no', 0.03, 0.45, 0.01, 0.35, 0.14, f0max)
    pointProcess = call(sound, "To PointProcess (periodic, cc)", f0min, f0max)
    formants = call(sound, "To Formant (burg)", 0.0025, 5, 5000, 0.025, 50)
    numPoints = call(pointProcess, "Get number of points")

    f1_list = []
    f2_list = []
    f3_list = []
    f4_list = []
    
    # Measure formants only at glottal pulses
    for point in range(0, numPoints):
        point += 1
        t = call(pointProcess, "Get time from index", point)
        f1 = call(formants, "Get value at time", 1, t, unit, 'Linear')
        f2 = call(formants, "Get value at time", 2, t, unit, 'Linear')
        f3 = call(formants, "Get value at time", 3, t, unit, 'Linear')
        f4 = call(formants, "Get value at time", 4, t, unit, 'Linear')
        f1_list.append(f1)
        f2_list.append(f2)
        f3_list.append(f3)
        f4_list.append(f4)
    
    f1_list = [f1 for f1 in f1_list if str(f1) != 'nan']
    f2_list = [f2 for f2 in f2_list if str(f2) != 'nan']
    f3_list = [f3 for f3 in f3_list if str(f3) != 'nan']
    f4_list = [f4 for f4 in f4_list if str(f4) != 'nan']
    
    # calculate mean formants across pulses
    if len(f1_list)!=0:
        f1_mean = statistics.mean(f1_list)
    else:
        f1_mean = 0
    if len(f2_list)!=0:
        f2_mean = statistics.mean(f2_list)
    else:
        f2_mean = 0
    if len(f3_list)!=0:
        f3_mean = statistics.mean(f3_list)
    else:
        f3_mean = 0
    if len(f4_list)!=0:
        f4_mean = statistics.mean(f4_list)
    else:
        f4_mean = 0
 
    
    # calculate median formants across pulses, this is what is used in all subsequent calcualtions
    # you can use mean if you want, just edit the code in the boxes below to replace median with mean
    if len(f1_list)!=0:
        f1_median = statistics.median(f1_list)
    else:
        f1_median = 0
    if len(f2_list)!=0:
        f2_median = statistics.median(f2_list)
    else:
        f2_median = 0
    if len(f3_list)!=0:
        f3_median = statistics.median(f3_list)
    else:
        f3_median = 0
    if len(f4_list)!=0:
        f4_median = statistics.median(f4_list)
    else:
        f4_median = 0

    
    return f1_mean, f2_mean, f3_mean, f4_mean, f1_median, f2_median, f3_median, f4_median

#9-Detection of Bulbar Involvement in Patients With Amyotrophic Lateral Sclerosis by Machine Learning Voice Analysis Diagnostic 
def measureFormants2(sound, wave_file,f0min,f0max,start,end,unit):
    dur10 = ((end - start)/10)
    midpoint = (start + (end - start)/2)
    start2 = midpoint - dur10
    end2 = midpoint + dur10

    formants = call(sound, "To Formant (burg)", 0.0025, 5, 5000, 0.025, 50)

    f1_mean = call(formants, "Get mean", 1, start2, end2, unit)
    f2_mean = call(formants, "Get mean", 2, start2, end2, unit)
    f3_mean = call(formants, "Get mean", 3, start2, end2, unit)
    f4_mean = call(formants, "Get mean", 4, start2, end2, unit)


    
    return f1_mean, f2_mean, f3_mean, f4_mean


def runPCA(df):
    # z-score the Jitter and Shimmer measurements
    measures = ['localJitter', 'localabsoluteJitter', 'rapJitter', 'ppq5Jitter', 'ddpJitter',
                'localShimmer', 'localdbShimmer', 'apq3Shimmer', 'apq5Shimmer', 'apq11Shimmer', 'ddaShimmer']
    x = df.loc[:, measures].values
    x = StandardScaler().fit_transform(x)
    # PCA
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(x)
    principalDf = pd.DataFrame(data = principalComponents, columns = ['JitterPCA', 'ShimmerPCA'])
    principalDf
    return principalDf


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
f1_median_list = []
f2_median_list = []
f3_median_list = []
f4_median_list = []
sex_list = []
healthy_list = []


# Go through all the wave files in the folder and measure all the acoustics
start = 0.0
end = 1.0
timeStep = 0.0
unit= "Hertz"

meta = pd.read_excel('../bd/SVD/META/SVD.xlsx', sheet_name='SVD')
ids = meta['ID'].tolist()
healthy = meta['Healthy'].tolist()
sex = meta['Sex'].tolist()
for wave_file in glob.glob("../bd/SVD/BD/A_NEUTRAL/*.wav"):
    
    x = wave_file.replace("../bd/SVD/BD/A_NEUTRAL/", "")
    id = x.replace("-a_n.wav", "")
    print(id)
    placeId = ids.index(int(id))

    sex_list.append(0 if sex[placeId]=="m" else 1)
    healthy_list.append(0 if healthy[placeId]=="n" else 1)
    f0min = 60 if sex[placeId]=="m" else 100
    f0max = 300 if sex[placeId]=="m" else 500

    sound = parselmouth.Sound(wave_file)
    (duration, meanF0, stdevF0, meanHNR, stdevHNR, localJitter, localabsoluteJitter, rapJitter, ppq5Jitter, ddpJitter, 
    localShimmer, localdbShimmer, apq3Shimmer, aqpq5Shimmer, apq11Shimmer, ddaShimmer) = measurePitch(sound, f0min, f0max, unit, start, end, timeStep)
    (f1_mean, f2_mean, f3_mean, f4_mean, f1_median, f2_median, f3_median, f4_median) = measureFormants1(sound, wave_file, f0min, f0max, unit)
    
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
    f1_median_list.append(f1_median)
    f2_median_list.append(f2_median)
    f3_median_list.append(f3_median)
    f4_median_list.append(f4_median)




# Add the data to Pandas
df = pd.DataFrame(np.column_stack([file_list, healthy_list, sex_list, mean_F0_list, sd_F0_list, meanHNR_list, stdevHNR_list, 
                                   localJitter_list, localabsoluteJitter_list, rapJitter_list, 
                                   ppq5Jitter_list, ddpJitter_list, localShimmer_list, 
                                   localdbShimmer_list, apq3Shimmer_list, aqpq5Shimmer_list, 
                                   apq11Shimmer_list, ddaShimmer_list, f1_mean_list, 
                                   f2_mean_list, f3_mean_list, f4_mean_list, 
                                   f1_median_list, f2_median_list, f3_median_list, 
                                   f4_median_list]),
                                   columns=['voiceID', 'healthy','sex', 'meanF0Hz', 'stdevF0Hz', 'meanHNR', 'stdevHNR', 
                                            'localJitter', 'localabsoluteJitter', 'rapJitter', 
                                            'ppq5Jitter', 'ddpJitter', 'localShimmer', 
                                            'localdbShimmer', 'apq3Shimmer', 'apq5Shimmer', 
                                            'apq11Shimmer', 'ddaShimmer', 'f1_mean', 'f2_mean', 
                                            'f3_mean', 'f4_mean', 'f1_median', 
                                            'f2_median', 'f3_median', 'f4_median'])

pcaData = runPCA(df) # Run jitter and shimmer PCA
df = pd.concat([df], axis=1) # Add PCA data

df2 = df.dropna(axis=0 , how='any')
df2.to_csv("X.csv", index=False)
