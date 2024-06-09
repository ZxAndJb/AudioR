from utils.waveuntils import wave
import numpy as np


class TextTemplate():
    def __init__(self, template, seed):
        self.template = template
        self.seed = seed

    def generatetext(self, w: wave):
        frequency_l = len(w.frequency)
        trend = self.checktrend(w.frequency)
        if frequency_l == 1:
            trend = "1"
            t_len = len(self.template[trend])
            pick = self.generateint(t_len)
            text = self.template[trend][pick].format(*w.frequency)

        else:
            if frequency_l > 4:
                raise ValueError("The maximum length of frequency is 4")
            else:
                t_len = len(self.template["{}".format(frequency_l)][trend])
                pick = self.generateint(t_len)
                text = self.template["{}".format(frequency_l)][trend][pick].format(*w.frequency)

        return text

    def generateint(self, t_len):
        return np.random.randint(0, t_len)

    def checktrend(self, frequencies: list):
        trend = ""
        for i in range(len(frequencies) - 1):
            if frequencies[i] < frequencies[i + 1]:
                trend += "u"
            else:
                trend += "d"
        return trend