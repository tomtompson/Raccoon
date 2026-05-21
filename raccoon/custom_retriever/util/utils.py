from pathlib import Path
import json
import numpy as np
import os
import torch
import pickle
from typing import List, Tuple, Dict
from collections import defaultdict
import re

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

def tokenize(text: str) -> List[str]:
    text = text.lower()
    text = re.sub(r"[^\w\sÀ-ÿ-]", " ", text, flags=re.UNICODE)
    return [t for t in text.split() if len(t) > 1]

def join_title_text(doc: dict) -> str:
    title = (doc.get("title") or "").strip()
    text = (doc.get("text") or "").strip()
    if title and text:
        return f"{title}\n{text}"
    return title or text

def rrf_fuse(rankings: List[Tuple[List[str], float]], k: int = 60) -> Dict[str, float]:
    scores = defaultdict(float)
    for ranking, weight in rankings:
        seen = set()
        for rank, doc_id in enumerate(ranking, start=1):
            if doc_id in seen:
                continue
            seen.add(doc_id)
            scores[doc_id] += weight / (k + rank)
    return dict(sorted(scores.items(), key=lambda x: x[1], reverse=True))

def min_max_normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.size == 0:
        return x
    mn, mx = x.min(), x.max()
    if mx - mn < 1e-12:
        return np.ones_like(x)
    return (x - mn) / (mx - mn)

def pretty_print_dict(d):
    pretty_dict = ''  
    
    for k, v in d.items():
        pretty_dict += f'{k}: \n'
        for value in v:
            pretty_dict += f'    {value}: {v[value]}\n'
    return pretty_dict


def deep_merge_dict(base: dict, new: dict) -> dict:
    for key, value in new.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            deep_merge_dict(base[key], value)
        else:
            base[key] = value
    return base


def append_results(file_path: str | Path, name: str, *args) -> None:
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    if file_path.exists():
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError:
            print(f"Warning: {file_path} was invalid JSON. Resetting it.")
            data = {}
    else:
        data = {}

    if name not in data:
        data[name] = {}

    for arg in args:
        if isinstance(arg, dict):
            deep_merge_dict(data[name], arg)
        else:
            print(f"Warning: Argument {arg} is not a dictionary and will be skipped.")

    tmp_file = file_path.with_suffix(file_path.suffix + ".tmp")

    with open(tmp_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)

    os.replace(tmp_file, file_path)

    print(f"Results appended to {file_path}")