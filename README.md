# Can You Learn Semantics Through Next-Word Prediction? The Case of Entailment

This repo contains the code and data that we used in our paper, [Can You Learn Semantics Through Next-Word Prediction? The Case of Entailment](https://arxiv.org/abs/2402.13956). We attempted to clean the repository up a bit from the one we used. All the functionality should be identical, but please reach out if you run into any issue!

## Environment

We specify the main library versions we used in `requirements.txt`. Nevertheless, during the process of the project, we occasionally updated the library versions (for example, to `transformers==4.33.3` and `accelerate==0.23.0` for Llama-2 support). So you may get slightly different results, but shouldn't by much.

## Data

### Broad-Coverage Entailment Datasets

The following command downloads and formats the RTE dataset to the format that we use. You can change it to process other datsets/splits.
```bash
python write_nli_data.py rte train datasets/rte_train.txt
```

### Generate Targeted Eval Data

The following command generates the passive targeted evaluation dataset. The other datasets work similarly. The `datasets/` directory also contains the generated datasets.
```bash
python targeted_entailment/passives.py datasets/passives.txt --pairs_path datasets/passives_pairs.txt
```

## Experiments

Use the following command to run the main experiment.
```bash
python entailment_test.py datasets/rte_train.txt rte_gpt2.pkl --model_name gpt2 --bsz 1
```

And the following command is an example to run the analysis where the premise is repeated.
```bash
python entailment_test.py datasets/rte_train.txt rte_gpt2_repeated.pkl --model_name gpt2 --bsz 1 --lhs_repetition 5 --rhs_repetition 5 --tie_repetitions
```

To compute the bits-per-byte (BPB) of a model on the C4 validation set, use for example:
```bash
python compute_perplexity.py gpt2 c4 en validation
```

And the following command trains a logistic regression classifier to predict the entailment label based on the sentence probabilities. Note that it expects the output pickle from the `entailment_test.py` script.
```bash
python trained_classifier.py datasets/rte_train.txt rte_gpt2.pkl
```
