# Problem Statement: Neural Machine Translation (English → Urdu)

## Overview

This is a research-based assignment in which students are required to design, implement, train, evaluate, and analyze a Neural Machine Translation (NMT) system using a **simple vanilla Recurrent Neural Network (RNN) architecture only** (no LSTM, GRU, or Transformer components).

## Objective

Build an English-to-Urdu translation system using a parallel corpus dataset (approximately 9,103 sentence pairs as provided).

**Dataset:**
- **Source:** Kaggle Translation Dataset
- **Link:** [https://www.kaggle.com/datasets/muhammadnoman76/translation-dataset](https://www.kaggle.com/datasets/muhammadnoman76/translation-dataset)
- **Relative Path:** `data/english_to_urdu_dataset.xlsx`
- **Format:** Parallel sentence pairs with columns `eng` (English) and `urdu` (Urdu)

## Deliverables

Students are expected to conduct experiments, analyze results, and document findings in a structured **LNCS-format technical report**.

## Required Tasks

### 1. Data Preprocessing

Download the dataset, load it into the development environment, inspect its structure, and design preprocessing pipelines for both English and Urdu sentences. This includes:

- Normalization of punctuation and whitespace
- Removal of corrupted samples
- Proper preparation of cleaned datasets

**Deliverables:**
- Display total pairs for both languages
- Show 5 random samples for both pairs
- Identify and fix any missing or duplicate records
- Save cleaned dataset as CSV

### 2. Train–Validation–Test Split

Create reproducible dataset splits (e.g., 80/10/10) using a fixed random seed.

**Deliverables:**
- Report the number of samples in each split
- Confirm that **no overlap exists** between partitions (especially on source sentences)
- Save split CSVs

### 3. Tokenization and Vocabulary Construction

Implement a word-level tokenization strategy for both languages and construct vocabularies including special tokens:

- **Padding token** (`<PAD>`)
- **Beginning-of-sentence token** (`<BOS>`)
- **End-of-sentence token** (`<EOS>`)

**Deliverables:**
- Document tokenization method
- Report final vocabulary sizes for source and target languages
- Save vocabulary objects (e.g., as pickle files)

### 4. Sequence Encoding, Padding, and Batching

Convert tokenized sentences into integer representations, apply padding, generate masks for padded tokens, and implement batching pipelines.

**Deliverables:**
- Correctly prepare encoder inputs, decoder inputs, and shifted target outputs for training
- Show evidence of working batch generation (tensor shapes or sample batch output)
- Implement masking or ignore-index handling for padded tokens

### 5. Vanilla RNN Encoder–Decoder Model Implementation

Design and implement an encoder–decoder architecture using **vanilla RNN layers only**.

**Deliverables:**
- Encoder implemented using vanilla RNN layers only
- Decoder implemented using vanilla RNN layers only
- Suitable selection of latent code space dimensionality
- Display summary of the architecture
- Report number of learnable parameters

### 6. Model Training and Experiment Tracking

Implement a complete training pipeline including:

- Loss computation
- Optimization
- **Teacher forcing** (must be implemented)
- Gradient clipping
- Checkpoint saving based on validation performance
- Selection of suitable hyperparameters
- Monitoring of training/validation loss

**Deliverables:**
- Training and validation loss curves properly displayed
- Convergence behavior discussed (overfitting, underfitting, etc.)
- Model checkpoints saved
- Training epochs set appropriately

### 7. Hyperparameter Tuning and Experimental Study

Conduct systematic hyperparameter tuning using **Grid Search**. Explore parameters such as:

- Embedding size
- Hidden dimension
- Number of RNN layers
- Learning rate
- Dropout
- Batch size

**Deliverables:**
- Multiple hyperparameter configurations tested
- Experimental comparison table showing performance across configurations
- Final selected hyperparameter configuration clearly identified
- Table showing: hyperparameter name, searched range, and optimal value selected

### 8. Inference, Decoding, and Evaluation

Implement inference using:

- **Greedy decoding** (required)
- **Beam search decoding** (at least one advanced method)

**Deliverables:**
- Evaluate the trained model on the test set using **BLEU score**
- Present representative translation examples (at least 10)
- Summarize results in a table (BLEU-1, BLEU-2, BLEU-4 for both decoding methods)
- Compare performance between greedy and beam search

### 9. Error Analysis and Research Discussion

Perform qualitative error analysis by manually evaluating at least **30 translated sentences** (as implemented in the notebook).

**Deliverables:**
- Identify common failure patterns
- Clear categorization of common translation error types
- Examples of strong and weak translations included
- Discussion of limitations of vanilla RNN–based translation models
- Discussion of possible future improvements

## Technical Constraints

- **Architecture:** Vanilla RNN only (no LSTM, GRU, or Transformer)
- **Hardware:** Optimized for RTX 4060 Laptop GPU with 8 GB VRAM
- **Dataset:** English–Urdu parallel corpus (~9,103 pairs as provided)

## Evaluation Criteria

The implementation will be evaluated based on:

1. Correctness of preprocessing and data handling
2. Proper train/val/test split with no overlap
3. Correct implementation of vanilla RNN encoder–decoder
4. Quality of hyperparameter tuning and experimental design
5. BLEU score performance on test set
6. Depth of error analysis and research discussion
7. Code quality, documentation, and reproducibility

## Notes

- All code should be well-documented and reproducible
- Use fixed random seeds for reproducibility
- Save all intermediate outputs (cleaned data, splits, vocabularies, checkpoints, results)
- Generate visualizations for training curves, BLEU scores, and error analysis
- The final report should follow LNCS format guidelines
