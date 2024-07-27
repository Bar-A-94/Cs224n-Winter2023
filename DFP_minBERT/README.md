# MultitaskBERT for Sentiment Analysis, Paraphrase Detection, and Semantic Textual Similarity

## Introduction

This project implements a MultitaskBERT model for three natural language processing tasks:

1. Sentiment Analysis (SST dataset)
2. Paraphrase Detection (PARA dataset)
3. Semantic Textual Similarity (STS dataset)

The project is based on the CS224N course at Stanford University, focusing on implementing and improving a BERT-based
model for multiple NLP tasks.

## Data

The model is trained and evaluated on three datasets:

- SST (Stanford Sentiment Treebank): Used for sentiment analysis - Contains 11,855 single sentences from movie reviews.
- PARA: Used for paraphrase detection - Consists of 400,000 question pairs with labels indicating whether particular
  instances are paraphrases of one another.
- STS (Semantic Textual Similarity): Used for measuring semantic similarity between sentence pairs - Consists of 8,628
  different sentence pairs of varying similarity on a scale from 0 (unrelated) to 5 (equivalent meaning).

## Approach

The approach for this project consists of three main objectives:

1. Implement minBERT
2. Analyze Sentiment with minBERT
3. Extend and Improve minBERT for Additional Downstream Tasks

### Implement minBERT

I implemented minBERT, incorporating key features of the original BERT model, including multi-head self-attention and
Transformer layers.

### Analyze Sentiment with minBERT

Evaluated minBERT on two downstream NLP tasks:

- Sentiment analysis on the Stanford Sentiment Treebank (SST)
- Sentiment analysis on the CFIMDB movie reviews database

This phase involved implementing the classification layer for both tasks and the Adam Optimizer algorithm.

### Extend and Improve minBERT for Additional Downstream Tasks

Extended the minBERT model to perform three tasks:

1. Sentiment analysis on the SST dataset (SST): multi-class classification
2. Paraphrase detection on the Quora dataset (Para): binary classification
3. Semantic Textual Similarity on the SemEval dataset (STS): regression

To improve the model's performance across these tasks, I implemented several extensions:

1. Learning Rate Scheduler:
    - Implemented a ReduceLROnPlateau scheduler
    - Dynamically adjusts the learning rate during training

2. Loss Function Optimization for STS:
    - Experimented with L1 loss, MSE, and Huber loss
    - Aimed to find the most effective approach for the regression task

3. STS Prediction Techniques:
    - Explored dot product and cosine similarity methods
    - Improved the model's ability to capture semantic relationships between sentences

4. Cosine Embedding Loss Pre-training:
    - Implemented a pre-training phase using cosine embedding loss
    - Enhanced the model's ability to capture semantic similarities across tasks

5. BERT Output Modifications:
    - Experimented with various BERT output representations:
      a. Using pooler output
      b. Using average of the last hidden layer
      c. Adding an interim layer
      d. Concatenating pooler output with the average last hidden state

6. Multi-stage Training Process:
    - Developed a three-stage training process:
      a. Full train cosine embedding loss with last layer
      b. Full train on the pooler output
      c. Fine-tune the entire model

7. Hyperparameter Tuning:
    - Conducted extensive tuning of learning rates, batch sizes, and model architectures
    - Optimized performance across all three tasks

These extensions were motivated by challenges encountered during initial prototyping and informed by a review of
relevant NLP literature. The goal was to create a robust multitask model capable of performing well across all three
downstream tasks.

## Experiments

### Baseline

- Basic model fine-tuned separately for each task, without a scheduler
- Results on dev set:
    - STS: 0.277
    - SST: 0.411
    - PARA: 0.701

### Improvements and Results

1. Learning Rate Scheduler:
    - Implemented ReduceLROnPlateau scheduler
    - Improved SST performance to 0.416

2. Loss Function for STS:
    - Experimented with L1 loss, MSE, and Huber loss
    - Best performance with L1 loss: 0.256

3. STS Prediction Techniques:
    - Tested dot product and cosine similarity
    - Best performance with cosine similarity (r=128): 0.237

4. Cosine Embedding Loss:
    - Pre-training with cosine embedding loss improved performance
    - STS (cosine similarity): 0.453, SST: 0.409, PARA: 0.706

5. BERT Output Modifications:
    - Experimented with different BERT output representations
    - Results:
      a. Using pooler output (baseline):
      STS: 0.277, SST: 0.411, PARA: 0.701
    -
   b. Using average of the last hidden layer:
   STS: 0.588, SST: 0.396, PARA: 0.690
    -
   c. Adding an interim layer for the BERT model:
   STS: 0.454, SST: 0.282, PARA: 0.640
    -
   **d. Concatenating pooler output with the average last hidden state and feed-forward:
   STS: 0.461, SST: 0.442, PARA: 0.717**

6. Multi-stage Training:
    - Developed a multi-stage training process:
      a. Full train cosine embedding loss with last layer
      b. Full train on the pooler output
      c. Fine-tune the entire model
    - Final results: STS: 0.530, SST: 0.385, PARA: 0.701
    - This approach showed a good balance between STS performance and maintaining competence in SST and PARA tasks

## Analysis

The experiments reveal several key insights:

1. The choice of loss function significantly impacts STS performance
2. Cosine similarity outperforms dot product for STS prediction
3. Pre-training with cosine embedding loss enhances the model's ability to capture semantic similarities
4. Combining different BERT output representations (pooler and last hidden state) leads to better overall performance
5. Multi-stage training, involving pre-training and fine-tuning, results in a good balance across all three tasks

## Conclusion

This project demonstrates the effectiveness of a MultitaskBERT model for sentiment analysis, paraphrase detection, and
semantic textual similarity tasks. Through various experiments and modifications, I achieved significant improvements
over the baseline model.

Best-performing model, which concatenates the pooler output with the average last hidden state and uses a feed-forward
layer, achieved the following results:

- STS: 0.461
- SST: 0.442
- PARA: 0.717

Compared to baseline results:

- STS: 0.277
- SST: 0.411
- PARA: 0.701

This represents substantial improvements across all tasks:

- STS: 66.4% improvement
- SST: 7.5% improvement
- PARA: 2.3% improvement

The most notable improvement is in the STS task, where our model significantly outperformed the baseline. While the
improvements in SST and PARA tasks are more modest, they still represent meaningful gains in performance.

The final model shows balanced performance across all three tasks, with particularly strong results in the STS and PARA
tasks. This demonstrates the potential of multitask learning in improving model generalization and performance across
diverse NLP tasks.

Future work could focus on further optimizing the multi-task learning process, exploring additional pre-training
techniques, and investigating ways to boost performance on the SST task while maintaining the strong results on STS and
PARA tasks.