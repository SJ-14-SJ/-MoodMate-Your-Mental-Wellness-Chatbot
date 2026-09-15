# MoodMate — Intent Classification Chatbot

An educational command-line chatbot built with PyTorch and NLTK. It classifies input into a predefined intent and selects a response from `intents.json`.

## Current implementation

- Tokenization, Porter stemming and bag-of-words features.
- A feed-forward neural network for intent classification.
- A confidence threshold of 0.75 before selecting a response.
- A fallback message when confidence is below the threshold.

The current repository does not include a Flask web interface, MongoDB persistence or sentiment tracking. Those are potential extensions.

## Setup and use

Use Python 3.12 and run from the repository root:

```bash
python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python chat.py
```

Type `quit` to exit. To retrain on the bundled intents:

```bash
python train.py
```

Training runs for 1,000 epochs and overwrites `data.pth`. Preserve the existing checkpoint if you want to compare models. Preprocessing uses NLTK tokenization with `preserve_line=True`, so it does not download sentence-tokenizer data. Training and inference share the same preprocessing and model definitions.

## Files

| File | Purpose |
| --- | --- |
| `intents.json` | Intent patterns and response templates |
| `train.py` | Training pipeline and checkpoint export |
| `model.py` | Inference model definition |
| `nltk_utils.py` | Text preprocessing and feature encoding |
| `chat.py` | Interactive terminal inference |
| `data.pth` | Existing model checkpoint |

## Limitations

This is a learning prototype, not a clinical tool or a crisis support service. It chooses predefined responses and cannot reliably understand a person's circumstances. Confidence scores are not validated measures of response safety or clinical suitability.

Training loss is not held-out accuracy. No held-out evaluation is currently supplied, and only CPU inference with the bundled checkpoint has been verified locally.

## Validation and portability

```bash
python -m unittest discover -s tests -v
```

The checkpoint is loaded onto CPU using restricted weights-only loading. Inference runs without gradient tracking. Empty input and entirely unknown vocabulary return a fallback before prediction. Tests cover CPU checkpoint loading, stemming, empty and unknown inputs, low-confidence fallback, and template response selection.

Training seeds NumPy and PyTorch for a reproducible baseline and saves CPU checkpoint tensors. GPU kernels can still vary across environments. Direct dependencies match the local Python 3.12 validation environment.

## Next improvements

- Add a held-out intent dataset, confusion matrix and failure analysis.
- Measure how response selection behaves on ambiguous or sensitive language.
- Add an interface only after clearly documenting the prototype's limitations.
