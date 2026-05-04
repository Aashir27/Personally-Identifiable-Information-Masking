# Personally Identifiable Information (PII) Masking

End-to-end notebooks for PII masking with two approaches:

- Fine-tuning DeBERTa for token classification (PER and EMAIL).
- Zero-shot extraction with LLaMA via OpenRouter and JSON parsing.

The workflow starts by augmenting the dataset with synthetic emails, then training and evaluating the model, and finally comparing results to a zero-shot baseline.

## Project Structure

- datasets/
	- data.json
	- test_data.json
	- train_data_modified.json
	- test_data_modified.json
- src/
	- prepare_data.ipynb
	- main.ipynb

## Notebooks

### prepare_data.ipynb

Creates augmented datasets by inserting synthetic emails into sequences. It:

- Loads data from datasets/data.json and datasets/test_data.json
- Extracts person names to generate realistic email addresses
- Inserts emails into token sequences with tags
- Saves the modified datasets to:
	- datasets/train_data_modified.json
	- datasets/test_data_modified.json

### main.ipynb

Runs DeBERTa fine-tuning and LLaMA zero-shot evaluation.

- DeBERTa model: microsoft/deberta-v3-base
- LLaMA model (default in notebook): meta-llama/llama-3.2-3b-instruct

The notebook:

- Loads the augmented datasets
- Fine-tunes DeBERTa for token classification
- Evaluates on validation and test sets (precision, recall, F1, accuracy)
- Computes per-entity FPR and FNR
- Runs zero-shot extraction with LLaMA and evaluates results

## Data Format

Each dataset entry is a JSON object with at least:

- tokens: list of token strings
- ner_tags: list of BIO tags aligned to tokens
- lang: language code (if present)
- sequence: original text (generated during augmentation)

Supported labels:

- O
- B-PER, I-PER
- B-EMAIL, I-EMAIL

## Setup

### Python Environment

Create a virtual environment and install dependencies:

```
pip install torch transformers datasets evaluate scikit-learn numpy openai python-dotenv
```

If you plan to use CUDA, install a matching GPU build of PyTorch.

### Model Paths

The notebook uses local model paths by default:

- D:/Coding/Models/DeBERTa-v3-base
- D:/Coding/Models/DeBERTa-v3-pii-masking
- D:/Coding/Models/DeBERTa-v3-pii-masking-final

Update these paths in main.ipynb or download the base model from Hugging Face and point to it.

### OpenRouter API Key

Zero-shot evaluation uses OpenRouter. Set this environment variable:

```
OPENROUTER_API_KEY=your_key_here
```

The notebook uses the OpenAI Python client with a custom base URL.

## Quickstart

1) Run src/prepare_data.ipynb to generate the augmented datasets.
2) Run src/main.ipynb to fine-tune DeBERTa and evaluate.
3) Compare DeBERTa vs LLaMA zero-shot metrics in the notebook output.

## Metrics Reported

- Overall token accuracy
- Precision, recall, F1 (micro and macro)
- Per-entity FPR (aggressive masking risk) and FNR (unmasked PII risk)

## Notes

- The zero-shot pipeline limits evaluation to the first 200 samples by default.
- You can switch the LLaMA model by changing the `model` variable in main.ipynb.
- CUDA and BF16 are enabled automatically when available.