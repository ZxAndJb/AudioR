import numpy as np
import scipy.io.wavfile as wav
from typing import Iterable
import librosa
from matplotlib import pyplot as plt
import itertools
from numpy import log
from numpy import asarray, pi
from scipy.signal import square, sawtooth
import re
import math

# %%

class wave:
    def __init__(self, wave, frequency: list, sr):
        self.sr = sr
        self.wave = wave
        self.frequency = frequency
        self.text_description = None

    def hann(self, window_size: int):
        return np.hanning(window_size)

    def concat(self, wavelist: Iterable, order: Iterable):
        window = self.hann(0.4 * self.sr)
        for i in range(len(wavelist)):
            index = math.floor(0.2 * self.sr)
            order_index = order[i]
            if order_index!= 0:
                wavelist[i].wave[:index] = wavelist[i].wave[:index] * window[:index]
            if order_index != len(order) - 1:
                wavelist[i].wave[-index:] = wavelist[i].wave[-index:] * window[index:]
        sorted_list = sorted(wavelist, key=lambda i: order[wavelist.index(i)])
        newwave = np.concatenate([e.wave for e in sorted_list])
        frequency_list = [w.frequency for w in sorted_list]
        frequencies = list(itertools.chain.from_iterable(frequency_list))
        return wave(newwave, frequencies, sr=self.sr)

    def superimpose(self, wavelist):
        container = np.zeros_like(wavelist[0].wave, dtype=np.float64)
        for i in range(len(wavelist)):
            container += wavelist[i].wave
        return wave(container, wavelist[0].frequency, sr=self.sr)

    def energy_scale_linear(self, scale_factor):
        assert scale_factor > 0, "scale factor must be a positive number"
        scale = np.linspace(1, scale_factor, self.wave.shape[0])
        return self.wave * scale

    def displayspectrogram(self, spectrogram: np.ndarray):
        librosa.display.specshow(spectrogram, sr=self.sr, y_axis='mel', hop_length=512, fmax=8820, x_axis='time')
        plt.title('Mel Spectrogram')
        plt.colorbar(format='%+2.0f dB')
        plt.show()  # 显示图表

    def spectrogram(self):
        mel_spect = librosa.feature.melspectrogram(y=self.wave, sr=self.sr, hop_length=512, window='hann', n_mels=128)
        mel_spect = librosa.power_to_db(mel_spect, ref=np.max)
        return mel_spect

    def plot_waveform(self):
        time = np.arange(0, len(self.wave)) * (1.0 / self.sr)
        plt.figure(figsize=(10, 4))  # 设置图形的尺寸
        plt.plot(time, self.wave)  # 绘制波形图
        plt.xlabel('Time')  # X轴标签
        plt.ylabel('Amplitude')  # Y轴标签
        plt.title('Waveform')  # 图表标题
        plt.grid(True)  # 显示网格
        plt.show()  # 显示图表

    def save(self, filename: str):
        wav.write(filename, self.sr, self.wave.astype(np.float32))


    def load(self, filename: str, text):
        sr, data = wav.read(filename)
        self.sr = sr
        self.wave = data
        self.text_description = text
        numbers = re.findall(r'\d+', text)
        self.frequency = numbers



class WaveGenerator():
    def __init__(self, samplerate):
        self.sr = samplerate
        self.period = 2 * pi

    def sawtooth(self, x, phase=None):
        if phase is not None:
            phase_offset = phase
        else:
            phase_offset = 0
        return 2 * ((x + phase_offset) / self.period - np.floor(0.5 + (x + phase_offset) / self.period))

    def Triangle(self, x, phase=None):
        if phase is not None:
            phase_offset = phase
        else:
            phase_offset = 0

        x = (x + phase_offset) % self.period
        if x <= self.period / 4:
            y = (4 / self.period) * x
        elif x > self.period / 4 and x <= self.period * 3 / 4:
            y = 2 - (4 / self.period) * x
        else:
            y = -4 + (4 / self.period) * x
        return y

    def square(self, x, phase=None):
        if phase is not None:
            phase_offset = phase
        else:
            phase_offset = 0
        return np.sign(np.sin(x + phase_offset))

    def sin(self, x, phase=None):
        if phase is not None:
            phase_offset = phase
        else:
            phase_offset = 0
        return np.sin(x + phase_offset)

    def cons(self, *args):
        return 1

    def generatePerWav(self, function, frequency, amplitude, duration, phase=None):
        if function == 'sin':
            waveform = self.sin
        elif function == 'square':
            waveform = self.square
        elif function == "sawtooth":
            waveform = self.sawtooth
        elif function == "triangle":
            waveform = self.Triangle
        elif function == "cons":
            waveform = self.cons
        else:
            raise ValueError("please input a valid function name")

        wavetable_length = 25600
        wavetable = np.zeros(wavetable_length)
        output = np.zeros((duration * self.sr))
        index = 0
        index_increment = frequency * wavetable_length / self.sr

        for n in range(wavetable_length):
            wavetable[n] = waveform(2 * np.pi * n / wavetable_length, phase) * amplitude

        for n in range(output.shape[0]):
            output[n] = self.interpolate_linearly(wavetable, index)
            index += index_increment
            index %= wavetable_length
        return output

    def interpolate_linearly(self, wavetable, indx):
        i = int(np.floor(indx))
        next_i = (i + 1) % wavetable.shape[0]
        left_weight = indx - i
        right_weight = 1 - left_weight

        return left_weight * wavetable[i] + right_weight * wavetable[next_i]

    def generatechirp(self, wavetype, t, f0, f1, t1, method, amplitude=1, phase=0):
        de = self.chirp_phase(t, f0, f1, t1, method)
        phase = phase * pi / 180
        if wavetype == "sin":
            return amplitude * np.sin(de + phase)
        elif wavetype == "sawtooth":
            return amplitude * sawtooth(de + phase)
        elif wavetype == "square":
            return amplitude * square(de + phase)
        else:
            return None

    def chirp_phase(self, t, f0, f1, t1, method='linear', vertex_zero=True):
        """
        Calculate the phase used by `chirp` to generate its output.

        See `chirp` for a description of the arguments.
        """
        t = asarray(t)
        f0 = float(f0)
        t1 = float(t1)
        f1 = float(f1)

        if method in ['linear', 'lin', 'li']:
            beta = (f1 - f0) / t1
            phase = 2 * pi * (f0 * t + 0.5 * beta * t * t)

        elif method in ['quadratic', 'quad', 'q']:
            beta = (f1 - f0) / (t1 ** 2)
            if vertex_zero:
                phase = 2 * pi * (f0 * t + beta * t ** 3 / 3)
            else:
                phase = 2 * pi * (f1 * t + beta * ((t1 - t) ** 3 - t1 ** 3) / 3)

        elif method in ['logarithmic', 'log', 'lo']:
            if f0 * f1 <= 0.0:
                raise ValueError("For a logarithmic chirp, f0 and f1 must be "
                                 "nonzero and have the same sign.")
            if f0 == f1:
                phase = 2 * pi * f0 * t
            else:
                beta = t1 / log(f1 / f0)
                phase = 2 * pi * beta * f0 * (pow(f1 / f0, t / t1) - 1.0)

        elif method in ['hyperbolic', 'hyp']:
            if f0 == 0 or f1 == 0:
                raise ValueError("For a hyperbolic chirp, f0 and f1 must be "
                                 "nonzero.")
            if f0 == f1:
                # Degenerate case: constant frequency.
                phase = 2 * pi * f0 * t
            else:
                # Singular point: the instantaneous frequency blows up
                # when t == sing.
                sing = -f1 * t1 / (f0 - f1)
                phase = 2 * pi * (-sing * f0) * log(np.abs(1 - t / sing))
        else:
            raise ValueError("method must be 'linear', 'quadratic', 'logarithmic',"
                             " or 'hyperbolic', but a value of %r was given."
                             % method)
        return phase

    def generateComposition(self, type, functions: list, frequency, amplitude: list, duration, phase, **kwargs):
        assert len(functions) == len(amplitude), "the length of functions and amplitude must be same"
        assert len(functions) <= 3 and len(functions) >= 1, "Compound 3 wave clips at most"
        wavedict = dict()
        for i in range(len(functions)):
            wavedict[i] = self.generate(type, functions[i], frequency, amplitude[i], duration, phase, **kwargs)
        newwave = wavedict[0].superimpose(list(wavedict.values()))
        return newwave

    def generate(self, type, functions, frequency: list, amplitude, duration, phase, **kwargs):
        if type == "chirp":
            method = kwargs["method"]
            if functions in ["sin", "sawtooth", "square"]:
                if len(frequency) != 2:
                    raise ValueError("The tuple should contain at least 2 numbers for frequency.")
                else:
                    t = np.linspace(0, duration, self.sr * duration)
                    wavedata = self.generatechirp(functions, t, frequency[0], frequency[1], duration, method, amplitude,
                                                  phase)
                    clip = wave(wavedata, frequency, self.sr)
            else:
                raise ValueError("function must be 'sin', 'sawtooth', 'square'")

        else:
            ## generate normal periodical wave clip
            wavedata = self.generatePerWav(functions, frequency[0], amplitude, duration, phase)
            clip = wave(wavedata, list(frequency), self.sr)
        return clip
