import os
from enum import Enum

import click
import librosa
import random
import numpy as np
import soundfile as sf
from tqdm import tqdm

import warnings
warnings.filterwarnings('ignore')

class TransformationType(Enum):
    FREQUENCY = "frequency"
    NOISE = "noise"
    IRREGULAR_NOISE = "irregular-noise"
    AMPLITUDE = "amplitude"


class Modification():
    def __init__(self, type, value_1, value_2=None):
        self.type = type
        self.value_1 = value_1
        self.value_2 = value_2

    def __str__(self) -> str:
        if self.value_2 is None:
            return f"{self.type}-{self.value_1}"
        return f"{self.type}-{self.value_1}-{self.value_2}"

    def __repr__(self) -> str:
        return self.__str__()


MODS = {
    TransformationType.FREQUENCY: [Modification("scale", 0.5), Modification("scale", 0.2), Modification("scale", 0.1)],
    TransformationType.NOISE: [Modification("avg", 0, 1), Modification("avg", 0, 10), Modification("avg", 5, 1)],
    TransformationType.IRREGULAR_NOISE: [Modification("irregular", "dog_bark")],
    TransformationType.AMPLITUDE: [Modification("amplitude", "randomly_chosen")]
}


class Transformation:
    def transform(self, y, sr):
        pass

    def __str__(self) -> str:
        return self.__class__.__name__

    def __repr__(self) -> str:
        return self.__str__()


class FrequencyTransformation(Transformation):
    def transform(self, y, sr, scale_factor=0.5):
        new_sr = int(sr * scale_factor)
        # print(f"Resampling from {sr} to {new_sr}")
        y = librosa.resample(y, orig_sr=sr, target_sr=new_sr)
        return y, new_sr

    def __str__(self) -> str:
        return TransformationType.FREQUENCY.value

    def __repr__(self) -> str:
        return self.__str__()


class NoiseTransformation(Transformation):
    def transform(self, y, sr, avg, std):
        noise = np.random.normal(avg, std, y.shape)
        y = y + noise
        return y, sr

    def __str__(self) -> str:
        return TransformationType.NOISE.value

    def __repr__(self) -> str:
        return self.__str__()


class IrregularNoiseTransformation(Transformation):
    def transform(self, y, sr):
        dog_bark_y, _ = librosa.load("source_files/dog-barking.mp3")
        y_max = np.max(y)
        dog_bark_y = dog_bark_y * y_max / 2 / np.max(dog_bark_y)
        y = y + dog_bark_y[:len(y)]
        return y, sr

    def __str__(self) -> str:
        return TransformationType.IRREGULAR_NOISE.value

    def __repr__(self) -> str:
        return self.__str__()

class AmplitudeTransformation(Transformation):
    def transform(self, y, sr):
        amplitude_factor = random.choice([25, 1, 0.04])
        # print(f"Changing amplitude with {amplitude_factor} factor")
        y = y * amplitude_factor
        return y, sr

    def __str__(self) -> str:
        return TransformationType.AMPLITUDE.value

    def __repr__(self) -> str:
        return self.__str__()

def file_generator(dir_path):
    files = []
    for root, _, files in os.walk(dir_path):
        for name in files:
            root_path = os.path.join(root, name)
            files.append((root_path, os.path.relpath(root_path, dir_path)))
    return files


def modify(data_dir, transfomation_type, output_dir):
    for data_file, rel_path in tqdm(file_generator(data_dir), desc="Processing files"):
        for mod in MODS[transfomation_type]:
            try:
                y, sr = librosa.load(data_file)
                if y is None:
                    continue
                if transfomation_type == TransformationType.FREQUENCY:
                    th = FrequencyTransformation()
                    y, sr = th.transform(y, sr, scale_factor=mod.value_1)
                elif transfomation_type == TransformationType.NOISE:
                    th = NoiseTransformation()
                    y, sr = th.transform(y, sr, mod.value_1, mod.value_2)
                elif transfomation_type == TransformationType.IRREGULAR_NOISE:
                    th = IrregularNoiseTransformation()
                    y, sr = th.transform(y, sr)
                elif transfomation_type == TransformationType.AMPLITUDE:
                    th = AmplitudeTransformation()
                    y, sr = th.transform(y, sr)
                else:
                    print("Unknown transformation")
            except Exception as e:
                print("Exception", e)
                continue
            output_file = os.path.join(output_dir, str(th), str(mod), rel_path)
            if not os.path.exists(os.path.dirname(output_file)):
                os.makedirs(os.path.dirname(output_file))
            # print(f"Writing to {output_file}")
            sf.write(output_file, y, sr, format='WAV')


@click.command()
@click.option('--data_dir', default="./data", required=True,
              help='Path to the data directory')
@click.option('--tr', default="frequency", required=True, help='Transformation type',
              type=click.Choice([t.value for t in TransformationType]))
@click.option('--output_dir', default="./data_mod",
              required=True, help='Path to the output directory')
def main(data_dir, tr, output_dir):
    print(
        f"Running with data_dir={data_dir}, transformation={tr}, output_dir={output_dir}")
    modify(data_dir, TransformationType(tr), output_dir)


main()
