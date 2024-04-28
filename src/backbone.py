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
    resp_objs = []
    if verbose:
        for audio in tqdm(audio_list):
            vector = model.extract_embedding(audio)
            if len(vector) < 512:
                resp_objs.append(vector[0])
            else:
                resp_objs.append(vector)
    else:
        for audio in audio_list:
            vector = model.extract_embedding(audio)
            if len(vector) < 512:
                resp_objs.append(vector[0])
            else:
                resp_objs.append(vector)
    vectors = [resp_obj["embedding"] for resp_obj in resp_objs]
    return vectors
