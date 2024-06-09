from torch.utils.data import Dataset
import scipy.io.wavfile as wav
import os
from transformers import AutoProcessor


processor = AutoProcessor.from_pretrained('laion/larger_clap_general')

class AudioTextDataset(Dataset):
    def __init__(self, datadf,audiopath):
        self.data = datadf
        self.audiopath = audiopath


    def __getitem__(self, index):
        item = self.data.iloc[index]
        filename = item["filename"]  ##MEl-spectrogram
        raw_text = item["text description"]
        file_path = os.path.join(self.audiopath, filename)
        sr, a_data = wav.read(file_path)
        return a_data, raw_text

    def __len__(self):
        return self.data.shape[0]

def collate_fn(batch):
    audio_batch, text_batch = [], []

    for a, t in batch:  # list of (item, audio_vec, text_vec)
        audio_batch.append(a)
        text_batch.append(t)

    input = process(audio_batch, text_batch, processor)

    return input

def process(audio, text, processor):
    return processor(text=text, audios=audio, return_tensors="pt", padding=True)

