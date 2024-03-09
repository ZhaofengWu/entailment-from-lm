import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizerFast, LlamaTokenizer


def max_model_length(tokenizer):
    if tokenizer.name_or_path.startswith("facebook/opt-"):
        # try running tokenizer.max_model_length. you will be amazed
        return 2048
    elif isinstance(tokenizer, (LlamaTokenizerFast, LlamaTokenizer)):
        # ditto
        return 2048  # https://github.com/facebookresearch/llama/issues/148#issuecomment-1459056594
    else:
        return (
            tokenizer.max_model_length
            if hasattr(tokenizer, "max_model_length")
            else tokenizer.model_max_length
        )


class EntailmentTester:
    def __init__(
        self,
        model_name: str,
        revision: str = None,
        lhs_repetition: int = 1,
        rhs_repetition: int = 1,
        tie_repetitions: bool = False,
        repeat_whitespace: bool = False,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            revision=revision,
            device_map="auto",
            offload_folder="offload_folder",
            torch_dtype="auto" if "llama-2" not in model_name.lower() else torch.bfloat16,
            offload_state_dict=True,
        ).eval()

        self.lhs_repetition = lhs_repetition
        self.rhs_repetition = rhs_repetition
        self.tie_repetitions = tie_repetitions
        self.repeat_whitespace = repeat_whitespace

    def prepare_input(self, sentence1: str, sentence2: str) -> tuple[str, str, str, str]:
        lhs_repetition = self.lhs_repetition
        rhs_repetition = self.rhs_repetition

        window = max_model_length(self.tokenizer) - len(self.tokenizer(sentence2).input_ids)
        if self.tie_repetitions:
            max_length = max(
                len(self.tokenizer(sentence1 + " ").input_ids),
                len(self.tokenizer(sentence2 + " ").input_ids),
            )
            max_lhs_repetition = max_rhs_repetition = min(
                window // max_length, lhs_repetition, rhs_repetition
            )
        else:
            lhs_length = len(self.tokenizer(sentence1 + " ").input_ids)
            max_lhs_repetition = min(window // lhs_length, lhs_repetition)

            rhs_length = len(self.tokenizer(sentence2 + " ").input_ids)
            max_rhs_repetition = min(window // rhs_length, rhs_repetition)

        lhs_repetition = min(lhs_repetition, max_lhs_repetition)
        rhs_repetition = min(rhs_repetition, max_rhs_repetition)

        prompt1 = self.get_prompt(sentence1, lhs_repetition)
        prompt2 = self.get_prompt(sentence2, rhs_repetition)
        # xy, x, yy, y
        return prompt1 + sentence2, prompt1[:-1], prompt2 + sentence2, prompt2[:-1]

    def get_prompt(self, sentence, n):
        prompt = (sentence + " ") * n
        if not self.repeat_whitespace:
            return prompt

        # just to be safe and make sure things are absolutely fair
        target_length = len(self.tokenizer(prompt).input_ids)
        prompt = sentence + " "
        single_length = len(self.tokenizer(prompt).input_ids)
        length_diff = target_length - single_length
        if isinstance(self.tokenizer, (LlamaTokenizerFast, LlamaTokenizer)):
            length_diff *= 16  # llama tokenizer considers 16 spaces as one token
        prompt += " " * length_diff
        assert len(self.tokenizer(prompt).input_ids) == target_length
        return prompt

    def prepare_inputs(
        self, sentences1: list[str], sentences2: list[str]
    ) -> tuple[list[str], list[str], list[str], list[str]]:
        prepared = []
        for sentence1, sentence2 in zip(sentences1, sentences2):
            prepared.append(self.prepare_input(sentence1, sentence2))
        return zip(*prepared)

    @torch.inference_mode()
    def get_log_probabilities(
        self, sentences: list[str], eos: bool = False, bsz: int = 1
    ) -> list[float]:
        all_input_ids = []
        for sentence in sentences:
            if eos:
                sentence = sentence + self.tokenizer.eos_token
            all_input_ids.append(self.tokenizer(sentence).input_ids)
        sorted_indices, all_input_ids = zip(
            *sorted(enumerate(all_input_ids), key=lambda x: -len(x[1]))
        )
        all_probs = []
        pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
        for i in tqdm(range(0, len(all_input_ids), bsz)):
            batch = all_input_ids[i : i + bsz]
            max_len = max(len(x) for x in batch)
            mask = [[1] * len(x) + [0] * (max_len - len(x)) for x in batch]
            input_ids = [x + [pad_token_id] * (max_len - len(x)) for x in batch]
            input_ids = torch.tensor(input_ids).cuda()
            mask = torch.tensor(mask).cuda()
            output = self.model(input_ids=input_ids, attention_mask=mask)
            shifted_logits = output.logits[:, :-1, :].log_softmax(-1)
            shifted_labels = input_ids[:, 1:]
            gold_logits = shifted_logits.gather(-1, shifted_labels.unsqueeze(-1)).squeeze(-1)
            probs = (gold_logits * mask[:, 1:]).sum(-1)
            all_probs.extend(probs.cpu().numpy().tolist())
        assert len(sorted_indices) == len(all_probs)

        unsorted_probs = [None] * len(all_probs)
        for i, prob in zip(sorted_indices, all_probs):
            unsorted_probs[i] = prob
        return unsorted_probs

    def test_entailment(
        self,
        sentences1: list[str],
        sentences2: list[str],
        bsz: int = 1,
    ) -> list[tuple[float, tuple[int, int, int, int], tuple[int, int]]]:
        lengths1 = [len(self.tokenizer(sentence).input_ids) for sentence in sentences1]
        lengths2 = [len(self.tokenizer(sentence).input_ids) for sentence in sentences2]
        xy, x, yy, y = self.prepare_inputs(sentences1, sentences2)
        logp_xy = self.get_log_probabilities(xy, bsz=bsz)
        logp_x = self.get_log_probabilities(x, eos=True, bsz=bsz)
        logp_yy = self.get_log_probabilities(yy, bsz=bsz)
        logp_y = self.get_log_probabilities(y, eos=True, bsz=bsz)
        return [
            (
                (logp_xy[i] - logp_x[i]) - (logp_yy[i] - logp_y[i]),
                (
                    logp_xy[i],
                    logp_x[i],
                    logp_yy[i],
                    logp_y[i],
                ),
                (
                    lengths1[i],
                    lengths2[i],
                ),
            )
            for i in range(len(sentences1))
        ]
