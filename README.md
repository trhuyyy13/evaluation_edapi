# DeepSeek Coder 1.3B Base Model Evaluation

This repository contains the stand-alone evaluation script and datasets required to benchmark the base `deepseek-ai/deepseek-coder-1.3b-base` model.

## Setup Instructions

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Download NLTK data (required for rouge and ngram entropy):
```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"
```

## Running Evaluation

To evaluate the base model, simply run the evaluation script:
```bash
python evaluate_base.py
```

The script will automatically:
1. Load `deepseek-ai/deepseek-coder-1.3b-base` directly from Hugging Face.
2. Load the eval data from `dataset/all.json`.
3. Run inference.
4. Calculate 4 assessment metrics (Efficacy, Generalization, Portability, Specificity).
5. Output final results to `results.json` and metric summaries to `mean_results.json`.
# evaluation_edapi
