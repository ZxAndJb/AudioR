import json
from numpy import random
import pandas as pd
import numpy as np
from waveuntils import wave, WaveGenerator
from TemplateGen import TextTemplate
import os
import hashlib



def roll_for_chirp(functions, methods, max_amplitude, max_frequency, max_phase, min_duration, max_duration):
    function = functions[np.random.randint(0, len(functions))]
    method = methods[np.random.randint(0, len(functions))]
    frequency_roll = list(np.random.randint(0, max_frequency, size=(2)))
    amplitude_roll = random.randint(0, max_amplitude + 1)
    phase_roll = random.randint(0, max_phase + 1)
    duration = random.randint(min_duration, max_duration + 1)
    return {
        "functions": function, "method": method, "frequency":frequency_roll, "amplitude":amplitude_roll,
        "phase":phase_roll, "duration":duration
    }

def roll_for_tone(functions, max_amplitude, max_frequency, max_phase, min_duration, max_duration):
    function = functions[np.random.randint(0, len(functions))]
    frequency_roll = [random.randint(0, max_frequency)]
    amplitude_roll = random.randint(0, max_amplitude + 1)
    phase_roll = random.randint(0, max_phase + 1)
    duration = random.randint(min_duration, max_duration + 1)
    return {
        "functions": function, "frequency":frequency_roll, "amplitude":amplitude_roll,
        "phase":phase_roll, "duration":duration
    }

def savewaves(wavelist, wave_path, csv_path):
    meta_df = pd.DataFrame(columns=["fid","filename","text description"])
    if not os.path.exists(wave_path):
        os.makedirs(wave_path)
    df_path = os.path.join(csv_path, "meta.csv")
    for idx, audio in enumerate(wavelist):
        filename = f"audio_{idx+1}.wav"
        file_path = os.path.join(wave_path, filename)
        audio.save(file_path)
        fid = generate_fixed_length_id(filename)
        meta_df.loc[meta_df.shape[0]] = [fid, filename, audio.text_description]
    meta_df.to_csv(df_path, index = False)


def generate_fixed_length_id(file_name, length=8):
    # 生成 SHA-256 哈希值
    hash_object = hashlib.sha256(file_name.encode())

    # 将哈希值转换为十六进制字符串
    hex_dig = hash_object.hexdigest()

    # 返回固定长度的哈希值
    return hex_dig[:length]

def generating(waveGenerator, textTemplate, amount, save_path):
    type = ["chirp", "Tone"]
    functions = {
        "chirp": ["sin", "sawtooth", "square"],
        "Tone": ["sin", "square", "sawtooth", "triangle", 'cons']
    }
    methods = ["linear", "quadratic", "logarithmic", "hyperbolic"]
    max_composition_time = 3
    max_amplitude = 100
    max_frequency = 4410
    max_phase = 360
    min_duration = 1
    max_duration = 3
    max_clips = 2
    waves = []
    for i in range(amount):
        clips = []
        print(f"this is {i}th audio")
        for j in range(max_clips):
            type_roll = type[np.random.randint(0, 2)]
            if_com = np.random.randint(0,2)
            if if_com:
                composition_time = random.randint(2, max_composition_time+1)
                if type_roll == "chirp":
                    main_meta = roll_for_chirp(functions[type_roll], methods, max_amplitude, max_frequency,
                                               max_phase, min_duration, max_duration)

                else:
                    main_meta = roll_for_tone(functions[type_roll], max_amplitude, max_frequency, max_phase,
                                              min_duration, max_duration)
                main_meta["functions"] = [main_meta["functions"]]
                main_meta["amplitude"] = [main_meta["amplitude"]]
                for k in range(composition_time-1):
                    main_meta["functions"].append(functions[type_roll][np.random.randint(0, len(functions))])
                    main_meta["amplitude"].append(random.randint(0, max_amplitude))
                wave = waveGenerator.generateComposition(type_roll,**main_meta)
                clips.append(wave)

            else:
                if type_roll == "chirp":
                    main_meta = roll_for_chirp(functions[type_roll], methods, max_amplitude, max_frequency,
                                               max_phase, min_duration, max_duration)

                else:
                    main_meta = roll_for_tone(functions[type_roll], max_amplitude, max_frequency, max_phase,
                                              min_duration, max_duration)
                wave = waveGenerator.generate(type_roll,**main_meta)
                clips.append(wave)
            print(f"this is the {j}-th clip for {i}-th audio, the clip is {if_com} composited, and the type is {type_roll},"
                  f"the component functions are {main_meta['functions']}, the frequency is {main_meta['frequency']},"
                  f"the amplitude is {main_meta['amplitude']}, the phase is {main_meta['phase']}, the method is ")
        order = list(range(0, len(clips)))
        random.shuffle(order)
        new_wave = wave.concat(clips,order)
        text = textTemplate.generatetext(new_wave)
        new_wave.text_description = text
        waves.append(new_wave)
        print("this is the assembled order: ", order)
        print(f"\rAudio_{i} has been created")


    savewaves(waves, save_path, save_path)


if __name__ == '__main__':
    with open("template.json", "r") as f:
        t = json.load(f)
    seed = np.random.seed(0)
    Tem = TextTemplate(t, seed)
    waveGenerator = WaveGenerator(13230)
    generating(waveGenerator, Tem, 5, "./data")
