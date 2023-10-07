import parselmouth
import pandas as pd
import os
import glob

# Amplitude normalization plus resampling

# filesPath: xxx
# fileSufix: xxx
# filesPathOutput: xxx
# metadataPath: xxx
# intensity: xxx
# resampling: xxx
def _NormalizeAmplitudeAndResampling(filesPath:str, filesPathReplace:str, filesPathOutput:str, intensity:int, resampling:int):
   for wave_file in glob.glob(filesPath+"/*.wav"):
    sound = parselmouth.Sound(wave_file)
    sound.scale_intensity(intensity)

    if resampling != 0:
        sound.resample(resampling)

    print(wave_file)
    name = wave_file.replace(filesPathReplace ,"")
    print(name)
    sound.save(filesPathOutput + name, 'WAV')



# Amplitude normalization plus resampling

# filesPath: xxx
# fileSufix: xxx
# filesPathOutput: xxx
# metadataPath: xxx
# intensity: xxx
# resampling: xxx
def _NormalizeAmplitudeAndResamplingAVFAD(filesPath:str, filesPathOutput:str, metadataPath:str, fileSufix:str, intensity:int, resampling:int):
    df = pd.DataFrame()
    meta = pd.read_excel(metadataPath, sheet_name='AVFAD')
    ids = meta['File ID'].tolist()
  

    for idx in ids:
            file = os.path.join(filesPath, str(idx)+"/"+str(idx)+fileSufix+".wav")
            print(filesPath, str(idx)+"/"+str(idx)+fileSufix+".wav")
            if(os.path.isfile(file)):
                sound = parselmouth.Sound(file)
                sound.scale_intensity(intensity)

                if resampling != 0:
                    sound.resample(resampling)

                sound.save(filesPathOutput+str(idx)+fileSufix+".wav", 'WAV')