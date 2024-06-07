from waveuntils import wave, WaveGenerator
import os
import glob
import pandas as pd
#%%

generator = WaveGenerator(13230)
dir_path = "./data"

wav_files = glob.glob(os.path.join(dir_path, '*.wav'))
meta_df = pd.read_csv(os.path.join(dir_path, 'meta.csv'))
wavdict = {}
for wav in wav_files:
    wavname = os.path.basename(wav)
    text = meta_df.loc[meta_df["filename"] == wavname, ['text description']]
    text = text['text description'].iloc[0]
    w = wave(None, [], "")
    w.load(wav, text)
    wavdict[wavname] = w

for n, w in wavdict.items():
    s = w.spectrogram()
    w.displayspectrogram(s)
    print(n)
    print(w.text_description)