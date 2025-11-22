# FSP: Focus Sentence Prompting for document-level MT Evaluation

FSP (Focus Sentence Prompting) is an LLM-as-a-Judge system for evaluating machine translation using the MQM framework. It uses large language models to perform segment-level MQM evaluation while preserving full document context, enabling more accurate long-form translation assessment.

FSP is based on the technique introduced in the paper ["Same evaluation, more tokens: On the effect of input length for machine translation evaluation using Large Language Models"](https://arxiv.org/pdf/2505.01761).

> 💡 FSP is a general-purpose document-level MT evaluation metric. Beyond the terminology-focused document-level tasks we evaluate in this work, it is broadly applicable to general document-level MT evaluation. Feel free to use it for your document-level evaluation tasks! 🚀

## Installation

### Required Dependencies

Install the following Python packages using `uv` (recommended):

```bash
uv pip install cohere nltk python-dotenv jinja2 openai pandas
```


### Environment Setup

Create a `.env` file in the `additional_metrics/fsp` directory with your API keys:

```bash
# For OpenAI models (GPT-4, GPT-4o, etc.)
OPENAI_API_KEY=your_openai_api_key_here

# For Cohere models (Command-A, etc.)
COHERE_API_KEY=your_cohere_api_key_here
```

## Running FSP Evaluation

### Quick Start

When located in the `wmt25-terminology/additional_metrics/fsp` directory, run evaluation scripts using:

```bash
# Example
bash scripts/gpt-5-2025-08-07-v2T-prompt-proper/run_CommandA_MT_evaluation.sh
```