import wespeaker
import numpy as np
from tqdm import tqdm


def build_representation(
    model,
    audio_list,
    method="wespeaker",
    verbose=False
):
    if method == "wespeaker":
        return build_representation_wespeaker(model, audio_list, verbose)
    else:
        raise NotImplementedError(f"Method {method} not implemented")


def build_representation_wespeaker(
    model,
    audio_list,
    verbose=False
):
    vectors = []
    if verbose:
        for audio in tqdm(audio_list, desc="Extracting embeddings"):
            vector = model.extract_embedding(audio)
            if len(vector) < 256:
                vectors.append(vector[0])
            else:
                vectors.append(vector)
    else:
        for audio in audio_list:
            vector = model.extract_embedding(audio)
            if len(vector) < 256:
                vectors.append(vector[0])
                print("vector invalid shape", vector.shape)
            else:
                vectors.append(vector)
    vectors = [np.array(vector) for vector in vectors]
    return vectors
