import os
import re
import ssl
import urllib.request

import numpy as np
import torch
import torch.nn.functional as F

ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
INVALID_ANS = "[invalid]"


def download_url(url: str, folder='folder'):
    """
    Downloads the content of an url to a folder. Modified from \
    https://github.com/pyg-team/pytorch_geometric/tree/master/torch_geometric

    Args:
        url (string): The url of target file.
        folder (string): The target folder.

    Returns:
        string: File path of downloaded files.
    """
    file = url.rpartition('/')[2]
    file = file if file[0] == '?' else file.split('?')[0]
    path = os.path.join(folder, file)
    if os.path.exists(path):
        print(f'File {file} exists, use existing file.')
        return path
    print(f'Downloading {url}')
    os.makedirs(folder, exist_ok=True)
    ctx = ssl._create_unverified_context()
    data = urllib.request.urlopen(url, context=ctx)
    with open(path, 'wb') as f:
        f.write(data.read())
    return path


def extract_answer_from_output(completion):
    match = ANS_RE.search(completion)
    if match:
        match_str = match.group(1).strip()
        match_str = match_str.replace(",", "")
        return match_str
    else:
        return INVALID_ANS


def is_correct(model_answer, answer):
    gt_answer = extract_answer_from_output(answer)
    assert gt_answer != INVALID_ANS
    return model_answer == gt_answer


def get_relative_top_filter(scores: torch.Tensor,
                            relative_top: float = 0.1,
                            min_tokens_to_keep: int = 1) -> torch.Tensor:
    scores_normalized: torch.Tensor = scores.log_softmax(dim=-1)
    sorted_logits, sorted_indices = torch.sort(scores_normalized,
                                               descending=True)
    min_thresh: torch.Tensor = sorted_logits[..., min_tokens_to_keep - 1]
    probs_max: torch.Tensor = torch.max(scores_normalized, dim=-1).values
    probs_thresh: torch.Tensor = probs_max + np.log(relative_top)
    probs_thresh = torch.min(min_thresh, probs_thresh)
    probs_thresh = probs_thresh.unsqueeze(-1)
    return torch.Tensor(scores_normalized < probs_thresh)
