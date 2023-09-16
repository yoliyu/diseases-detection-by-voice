
import pandas as pd
import glob
import numpy as np
import parselmouth 
import statistics
from parselmouth.praat import call
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

def _printDatasetReport(filePath):
    name_list = []
    total_list = []
    women_list = []
    men_list = []
    healthy_list = []
    pathological_list = []


    for file in glob.glob(filePath+"*.csv"):
        
        dataframe = pd.read_csv(file)
        name_list.append(file.replace(".csv",""))
        total_list.append(dataframe.shape[0])
        if "all" in file:
            y=dataframe['sex'].values
            women_list.append((y == 0).sum())
            men_list.append((y == 1).sum())

        else:
            women_list.append(0)
            men_list.append(0)
        
        y = dataframe['healthy'].values
        healthy_list.append((y == 0).sum())
        pathological_list.append((y == 1).sum())
    


    report = pd.DataFrame(np.column_stack([name_list, total_list,women_list,men_list, healthy_list, pathological_list]),
                                                columns=['name', 'total', 'women','men','healthy', 'pathological'])
    report.to_csv("report.csv")

def _plotDatasetReport(women_healthy,women_pathological,men_healthy, men_pathological,name):
    # Creating dataset
    size = 2
    cars = ['healthy', 'pathological']
    
    data = np.array([[women_healthy,women_pathological], [men_healthy, men_pathological]])
    
    # normalizing data to 2 pi
    norm = data / np.sum(data)*2 * np.pi
    
    # obtaining ordinates of bar edges
    left = np.cumsum(np.append(0,norm.flatten()[:-1])).reshape(data.shape)
    
    # Creating color scale
    cmap = plt.get_cmap("tab20c")
    outer_colors = cmap(np.array([13,16]))
    inner_colors = cmap(np.array([14, 15, 18, 19]))
    
    # Creating plot
    fig, ax = plt.subplots(figsize =(10, 7),
                        subplot_kw = dict(polar = True))
    
    ax.bar(x = left[:, 0],
        width = norm.sum(axis = 1),
        bottom = 1-size,
        height = size,
        color = outer_colors,
        edgecolor ='w',
        linewidth = 1,
        align ="edge")

    ax.bar(x = left.flatten(),
        width = norm.flatten(),
        bottom = 1-2 * size,
        height = size,
        color = inner_colors,
        edgecolor ='w',
        linewidth = 1,
        align ="edge")

    ax.set(title =name)
    ax.set_axis_off()

    # show plot
    plt.show()

# Crea una carpeta con espectrogramas
# filesPath: path donde se encuentran los ficheros a recorrer
# metadataPath: path donde están los metadatos
# fileSufix: sufijo de los ficheros que se van a recorrer en 01-phrase.wav, sería "phrase"
def _spectrogramCreation(filesPath:str, metadataPath:str, fileSufix:str, outputPath:str):
    meta = pd.read_excel(metadataPath, sheet_name='SVD')
    ids = meta['ID'].tolist()
    sex = meta['Sex'].tolist()
    for wave_file in glob.glob(filesPath+"*.wav"): 
        x = wave_file.replace(filesPath, "")
        id = x.replace("-"+fileSufix+".wav", "")
        id = id.replace("\\", "")
        placeId = ids.index(int(id))
        print(id)
        plt = _createSpectrogram(wave_file)
        plt.savefig(outputPath+"/women/"+id+".jpg") if sex[placeId]=="m" else plt.savefig(outputPath+"/men/"+id+".jpg")
        plt.close()

def _ageFilter(target,minAge,maxAge):
    return (minAge<=target)and(target<=maxAge)

# Create spectrogram by path of a file
def _createSpectrogram(filePath:str):
    dynamic_range=50
    sound = parselmouth.Sound(filePath)
    sound.pre_emphasize()
    spectrogram = sound.to_spectrogram(window_length=0.05, 
                                maximum_frequency=5500)
    X, Y = spectrogram.x_grid(), spectrogram.y_grid()
    sg_db = 10 * np.log10(spectrogram.values)
    
    fig = plt.figure(figsize=(16,12))
    plt.pcolormesh(X, Y, sg_db, vmin=sg_db.max() - dynamic_range, cmap='afmhot')
    plt.ylim([spectrogram.ymin, spectrogram.ymax])
    plt.gca().set_axis_off()
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0,
            hspace = 0, wspace = 0)
    plt.margins(0,0)
    fig.canvas.draw()
    return plt

# This is the function to measure source acoustics.
def _measurePitch(voiceID:str, f0min:int, f0max:int, unit:str, start:int, end:int, timeStep:int):
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
def _measureFormants1(sound:parselmouth.Sound,f0min:int,f0max:int,unit:str):
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
def _measureFormants2(sound:parselmouth.Sound,start:int,end:int,unit:str):
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

def _runPCA(df:pd.DataFrame):
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

def _mfcc(filePath:str):
    mfccs_array = []
    sound = parselmouth.Sound(filePath)
    duration = call(sound, "Get total duration") # duration
    print(duration)
    mfcc_object = sound.to_mfcc(number_of_coefficients=13,window_length=0.25,time_step=0.25) 
    mfccs = mfcc_object.to_array()
    for iy, ix in np.ndindex(mfccs.shape):
        mfccs_array.append(mfccs[iy, ix])
    x = pd.DataFrame(np.column_stack([mfccs_array]))
    return x