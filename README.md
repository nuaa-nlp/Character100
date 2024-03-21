<div align="center">
    <h1>Character100 : A Benchmark for Characteristic AI Agents</h1>
    <p>
      <a href="https://character100.github.io/">Project Page</a> - 
      <a href="https://arxiv.org/abs/2403.12368">Paper</a>
    </p>
</div>



This is the repository for COLING2024 paper "Characteristic AI Agents via Large Language Models". 

**Character100** is a comprehensive benchmark designed to evaluate and compare the performance of characteristic AI agents. This benchmark includes a dataset tailored for the task, along with automatic evaluation metrics to measure and compare the capabilities of AI agents.

# Dataset and Resources

## Raw Data Aquisition

The goal of raw data acquisition is to extract relevant information from Wikipedia pages. Pre-extracted data can be found in the `Data/raw_data` directory for your convenience. Alternatively, you can utilize the code `Code/get_raw_data.py` to perform your own data extraction.

## Data Organization

We have put the raw context-question-response pairs in `Data/QR`. These pairs are further divided into training (`Data/QR_train`) and testing (`Data/QR_test`). For the convenience, we also provide processed files (`Data/train.json`, `Data/dev.json`, and `Data/test.txt`) for direct use in training your models.

# Training

## Training of LLMs

You can use the above-mentioned data to train your own LLMs.

## Training of Discriminator

The origin style corpus for the 106 people is in `Data/interviews_origin`, and the balanced corpus is in `Data/interview_processed`. You can use `Data/discriminator_train.json` for training.

# Evaluation Metrics

## Background Knowledge Consistency

### BLEU and ROGUE

Process the results into predict file and truth file, and then use `Code/background_knowledge.py` to evaluate the BLEU and ROGUE score.

### Semantic Similarity

Use `Code/semantic.py` to calculate the BLEU and ROGUE score.

## Style Consistency

You can use `Code/discriminator_train.py` to train your discriminator, or you can use our trained [checkpoint](https://drive.google.com/drive/folders/1eQTA1-sp_bgFXWUHuYodvWcXrHQLiQph?usp=sharing).

Use `Code/style.py` to evaluate the style consistency score.

# Citation
```bibtex
@misc{wang2024characteristic,
      title={Characteristic AI Agents via Large Language Models}, 
      author={Xi Wang and Hongliang Dai and Shen Gao and Piji Li},
      year={2024},
      eprint={2403.12368},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
