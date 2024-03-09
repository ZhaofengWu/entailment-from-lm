import math
import sys

import datasets
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.entailment_tester import max_model_length


@torch.inference_mode()
def compute_bpb(tokenizer, model, sentences, bsz=1):
    num_tokens = 0
    num_bytes = 0
    all_input_ids = []
    max_len = max_model_length(tokenizer)
    for sentence in tqdm(sentences):
        input_ids = tokenizer(sentence, truncation=True, max_length=max_len).input_ids
        all_input_ids.append(input_ids)
        num_tokens += len(input_ids)
        num_bytes += len(sentence.encode("utf-8"))
    all_input_ids = sorted(all_input_ids, key=lambda x: -len(x))
    all_probs = []
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    for i in tqdm(range(0, len(all_input_ids), bsz)):
        batch = all_input_ids[i : i + bsz]
        max_len = max(len(x) for x in batch)
        mask = [[1] * len(x) + [0] * (max_len - len(x)) for x in batch]
        input_ids = [x + [pad_token_id] * (max_len - len(x)) for x in batch]
        input_ids = torch.tensor(input_ids).cuda()
        mask = torch.tensor(mask).cuda()
        output = model(input_ids=input_ids, attention_mask=mask)
        shifted_logits = output.logits[:, :-1, :].log_softmax(-1)
        shifted_labels = input_ids[:, 1:]
        gold_logits = shifted_logits.gather(-1, shifted_labels.unsqueeze(-1)).squeeze(-1)
        probs = (gold_logits * mask[:, 1:]).sum(-1)
        all_probs.extend(probs.cpu().numpy().tolist())
    assert len(all_input_ids) == len(all_probs)
    total_log_probs = sum(all_probs)
    total_tokens = sum(len(x) for x in all_input_ids)
    perp = np.exp(-total_log_probs / total_tokens)
    return math.log(perp) * (num_tokens / num_bytes) / math.log(2)


def main(model_name, dataset_name, subset_name, split_name):
    dataset = datasets.load_dataset(dataset_name, subset_name, split=split_name)
    sentences = dataset["text"]

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        offload_folder="offload_folder",
        torch_dtype="auto" if "llama-2" not in model_name.lower() else torch.bfloat16,
        offload_state_dict=True,
    ).eval()
    bpb = compute_bpb(tokenizer, model, sentences)
    print(bpb)


if __name__ == "__main__":
    main(*sys.argv[1:])  # pylint: disable=no-value-for-parameter
