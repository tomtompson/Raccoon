import numpy as np
import os
import torch
import pickle

def save_embeddings(
    embeddings: np.ndarray | list[torch.Tensor], text_ids: list[str], output_filename: str = "./embeddings/"
):
    """
    Saves the embeddings to a pickle file.
    :param embeddings: The embeddings to save.
    :param output_path: The path where the embeddings will be saved.
    """
    os.makedirs(os.path.dirname(output_filename), exist_ok=True)

    if isinstance(embeddings[0], torch.Tensor):
        embeddings = embeddings.cpu().detach().numpy()  # Convert to numpy array if it's a tensor

    with open(output_filename, "wb") as f:
        pickle.dump((embeddings, text_ids), f)


def pickle_load(path: str) -> tuple[np.ndarray, list[str]]:
    with open(path, "rb") as f:
        reps, lookup = pickle.load(f)
    return np.array(reps), lookup
