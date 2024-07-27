"""
Multitask BERT class, starter training code, evaluation, and test code.

Of note are:
* class MultitaskBERT: Your implementation of multitask BERT.
* function train_multitask: Training procedure for MultitaskBERT. Starter code
    copies training procedure from `classifier.py` (single-task SST).
* function test_multitask: Test procedure for MultitaskBERT. This function generates
    the required files for submission.

Running `python multitask_classifier.py` trains and tests your MultitaskBERT and
writes all required submission files.
"""

from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from bert import BertModel
from datasets import create_train_dataloaders, create_dev_dataloaders, create_test_dataloaders
from evaluation import model_eval_sst, model_eval_para, model_eval_sts, model_eval_multitask, model_eval_test_multitask
from optimizer import AdamW
from utils import save_model, print_and_log, seed_everything, get_args

TQDM_DISABLE = False
BERT_HIDDEN_SIZE = 768
N_SENTIMENT_CLASSES = 5


class MultitaskBERT(nn.Module):
    """
    This module should use BERT for 3 tasks:

    - Sentiment classification (predict_sentiment)
    - Paraphrase detection (predict_paraphrase)
    - Semantic Textual Similarity (predict_similarity)
    MultitaskBERT(
      (bert): BertModel(
        (word_embedding): Embedding(30522, 768, padding_idx=0)
        (pos_embedding): Embedding(512, 768)
        (tk_type_embedding): Embedding(2, 768)
        (embed_layer_norm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
        (embed_dropout): Dropout(p=0.1, inplace=False)
        (bert_layers): ModuleList(
          (0-11): 12 x BertLayer(
            (self_attention): BertSelfAttention(
              (query): Linear(in_features=768, out_features=768, bias=True)
              (key): Linear(in_features=768, out_features=768, bias=True)
              (value): Linear(in_features=768, out_features=768, bias=True)
              (dropout): Dropout(p=0.1, inplace=False)
            )
            (attention_dense): Linear(in_features=768, out_features=768, bias=True)
            (attention_layer_norm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
            (attention_dropout): Dropout(p=0.1, inplace=False)
            (interm_dense): Linear(in_features=768, out_features=3072, bias=True)
            (dropout): Dropout(p=0.1, inplace=False)
            (out_dense): Linear(in_features=3072, out_features=768, bias=True)
            (out_layer_norm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
            (out_dropout): Dropout(p=0.1, inplace=False)
          )
        )
        (pooler_dense): Linear(in_features=768, out_features=768, bias=True)
        (pooler_af): Tanh()
      )
      (dropout): Dropout(p=0.3, inplace=False)
      (mix_pooler_last_hidden): Linear(in_features=1536, out_features=768, bias=True)
      (sentiment_last_layer): Linear(in_features=768, out_features=5, bias=True)
      (paraphrase_last_layer): Linear(in_features=1536, out_features=1, bias=True)
      (similarity_layer): Linear(in_features=768, out_features=32, bias=True)
      (cos): CosineSimilarity()
      (sigmoid): Sigmoid()
    )


    """

    def __init__(self, config):
        """
            Initialize the MultitaskBERT model.

            Input:
            - config: A configuration object containing model parameters
                config = {'hidden_dropout_prob': args.hidden_dropout_prob,
                  'num_labels': num_labels,
                  'hidden_size': 768,
                  'data_dir': '.',
                  'fine_tune_mode': args.fine_tune_mode}

            The function initializes BERT and adds task-specific layers for each task.
        """
        super(MultitaskBERT, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.grad_change(config)
        # My code:
        self.r = 32
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.mix_pooler_last_hidden = nn.Linear(2 * config.hidden_size, config.hidden_size)
        self.sentiment_last_layer = nn.Linear(config.hidden_size, N_SENTIMENT_CLASSES)
        self.paraphrase_last_layer = nn.Linear(2 * config.hidden_size, 1)
        self.similarity_layer = nn.Linear(config.hidden_size, self.r)
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        self.sigmoid = nn.Sigmoid()
        self.gelu = F.gelu

    def grad_change(self, config, use_config=True):
        """
            Set whether BERT parameters should be fine-tuned or frozen.

            Inputs:
            - config: Configuration object
            - use_config: Boolean indicating whether to use the config for determining fine-tuning

            This function sets the requires_grad attribute of BERT parameters based on the fine-tuning mode.
        """
        if use_config:
            # last-linear-layer mode does not require updating BERT parameters.
            assert config.fine_tune_mode in ["last-linear-layer", "full-model"]
            for param in self.bert.parameters():
                if config.fine_tune_mode == 'last-linear-layer':
                    param.requires_grad = False
                elif config.fine_tune_mode == 'full-model':
                    param.requires_grad = True
        else:
            for param in self.bert.parameters():
                param.requires_grad = True

    def forward(self, input_ids, attention_mask):
        """
            Forward pass through BERT and additional layers.

            Inputs:
            - input_ids: Tensor of token ids
            - attention_mask: Tensor of attention mask

            Output:
            - bert_output: Tensor of processed BERT output

            This function processes the input through BERT and additional layers.
        """

        # Use the poller output
        pooler_output = self.bert(input_ids, attention_mask)['pooler_output']

        # Use the mean of the last hidden state
        mean_last_hidden = torch.mean(self.bert(input_ids, attention_mask)['last_hidden_state'], dim=1)

        # Use both for feed forward
        con = torch.cat((pooler_output, mean_last_hidden), dim=1)
        bert_output = self.mix_pooler_last_hidden(con)

        return bert_output

    def output_embeddings_last_hidden(self, input_ids_1, attention_mask_1, input_ids_2, attention_mask_2):
        """
            Output embeddings using the mean of the last hidden state.

            Inputs:
            - input_ids_1, input_ids_2: Tensor of token ids for two sentences
            - attention_mask_1, attention_mask_2: Tensor of attention masks for two sentences

            Outputs:
            - bert_output_1, bert_output_2: Tensors of embeddings for two sentences

            This function is used for processing sentence pairs, particularly for paraphrase and similarity tasks.
        """
        bert_output_1 = torch.mean(self.bert(input_ids_1, attention_mask_1)['last_hidden_state'], dim=1)
        bert_output_2 = torch.mean(self.bert(input_ids_2, attention_mask_2)['last_hidden_state'], dim=1)
        return bert_output_1, bert_output_2

    def output_embeddings_pooler(self, input_ids_1, attention_mask_1, input_ids_2, attention_mask_2):
        """
            Output embeddings using the pooler output.

            Inputs:
            - input_ids_1, input_ids_2: Tensor of token ids for two sentences
            - attention_mask_1, attention_mask_2: Tensor of attention masks for two sentences

            Outputs:
            - bert_output_1, bert_output_2: Tensors of embeddings for two sentences

            This function provides an alternative way of generating sentence embeddings.
        """
        bert_output_1 = self.bert(input_ids_1, attention_mask_1)['pooler_output']
        bert_output_2 = self.bert(input_ids_2, attention_mask_2)['pooler_output']
        return bert_output_1, bert_output_2

    def output_embeddings(self, input_ids_1, attention_mask_1, input_ids_2, attention_mask_2):
        """
            Output embeddings using the combined forward method.

            Inputs:
            - input_ids_1, input_ids_2: Tensor of token ids for two sentences
            - attention_mask_1, attention_mask_2: Tensor of attention masks for two sentences

            Outputs:
            - bert_output_1, bert_output_2: Tensors of embeddings for two sentences

            This function uses the custom forward method to generate embeddings.
        """
        bert_output_1 = self.forward(input_ids_1, attention_mask_1)
        bert_output_2 = self.forward(input_ids_2, attention_mask_2)
        return bert_output_1, bert_output_2

    def predict_sentiment(self, input_ids, attention_mask):
        """
            Predict sentiment (5 classes) for given input.

            Inputs:
            - input_ids: Tensor of token ids
            - attention_mask: Tensor of attention mask

            Output:
            - Tensor of logits for 5 sentiment classes

            This function is used for the sentiment classification task.
        """
        bert_out = self.forward(input_ids, attention_mask)
        return self.dropout(self.sentiment_last_layer(bert_out))

    def predict_paraphrase(self,
                           input_ids_1, attention_mask_1,
                           input_ids_2, attention_mask_2):
        """
            Predict whether two sentences are paraphrases.

            Inputs:
            - input_ids_1, input_ids_2: Tensor of token ids for two sentences
            - attention_mask_1, attention_mask_2: Tensor of attention masks for two sentences

            Output:
            - Tensor of logits for paraphrase prediction

            This function is used for the paraphrase detection task.
        """
        bert_out_1 = self.forward(input_ids_1, attention_mask_1)
        bert_out_2 = self.forward(input_ids_2, attention_mask_2)
        concat_embed = torch.cat((bert_out_1, bert_out_2), dim=1)
        return self.dropout(self.paraphrase_last_layer(concat_embed))

    def predict_similarity(self,
                           input_ids_1, attention_mask_1, input_ids_2, attention_mask_2):
        """
            Predict semantic similarity between two sentences.

            Inputs:
            - input_ids_1, input_ids_2: Tensor of token ids for two sentences
            - attention_mask_1, attention_mask_2: Tensor of attention masks for two sentences

            Output:
            - Tensor of similarity scores

            This function is used for the semantic textual similarity task.
        """
        bert_out_1 = self.dropout(self.gelu(self.similarity_layer(self.forward(input_ids_1, attention_mask_1))))
        bert_out_2 = self.dropout(self.gelu(self.similarity_layer(self.forward(input_ids_2, attention_mask_2))))

        # Cosine similarity:
        out = 5 * self.sigmoid(self.cos(bert_out_1, bert_out_2).unsqueeze(-1))

        # Dot product:
        # bert_out_1_shaped = bert_out_1.view(bert_out_1.shape[0], 1, bert_out_1.shape[1])
        # bert_out_2_shaped = bert_out_2.view(bert_out_1.shape[0], bert_out_1.shape[1], 1)
        # out = 5 * (bert_out_1_shaped @ bert_out_2_shaped).reshape(input_ids_1.shape[0], -1)

        return out


def train_multitask(args):
    """
        Main training function for MultitaskBERT.

        Input:
        - args: Command line arguments

        This function orchestrates the entire training process:
            1. Set up the device (GPU/CPU)
            2. Create dataloaders for all tasks
            3. Initialize the MultitaskBERT model
            4. Train the model using cosine embedding
            5. Fine-tune the model for each specific task
    """
    device = torch.device('mps') if args.use_gpu else torch.device('cpu')
    print("Using device: {}".format(device))
    # Create the data and its corresponding datasets and dataloader.
    sst_train_dataloader, para_train_dataloader, sts_train_dataloader = create_train_dataloaders(args)
    sst_dev_dataloader, para_dev_dataloader, sts_dev_data, sts_dev_dataloader, num_labels = create_dev_dataloaders(args)

    print("Init model")
    config = {'hidden_dropout_prob': args.hidden_dropout_prob,
              'num_labels': num_labels,
              'hidden_size': 768,
              'data_dir': '.',
              'fine_tune_mode': args.fine_tune_mode}

    config = SimpleNamespace(**config)

    model = MultitaskBERT(config)
    model = model.to(device)
    print("model saved to {}".format(device))
    with open(args.logpath, "w") as f:
        f.write("Training started\n")
    train_cosine_embedding(args, config, device, model, sts_train_dataloader, model.output_embeddings_last_hidden)
    train_cosine_embedding(args, config, device, model, sts_train_dataloader, model.output_embeddings_pooler)
    train_cosine_embedding(args, config, device, model, sts_train_dataloader, model.output_embeddings)
    for data_set_name, inputs, train_dataloader, dev_dataloader, predict_func, loss_func, eval_func in [
        ("sts", 2, sts_train_dataloader, sts_dev_dataloader, model.predict_similarity, F.l1_loss, model_eval_sts),
        ("sst", 1, sst_train_dataloader, sst_dev_dataloader, model.predict_sentiment, F.cross_entropy, model_eval_sst),
        ("para", 2, para_train_dataloader, para_dev_dataloader, model.predict_paraphrase,
         nn.BCEWithLogitsLoss(reduction="sum"), model_eval_para)]:
        print_and_log(args, "Start training for {} dataset".format(data_set_name))
        train(inputs, args, config, device, model, train_dataloader, dev_dataloader, predict_func, loss_func, eval_func)


def train_cosine_embedding(args, config, device, model, train_dataloader, predict_func):
    """
        Train the model using cosine embedding loss.

        Inputs:
        - args: Command line arguments
        - config: Model configuration
        - device: Device to run the model on
        - model: The MultitaskBERT model
        - train_dataloader: DataLoader for training data
        - predict_func: Function to generate embeddings

        This function trains the model to generate similar embeddings for similar sentences.
        The flow is:
        1. Set up optimizer and scheduler
        2. For each epoch:
            a. For each batch:
                - Generate embeddings
                - Compute loss
                - Backpropagate and update model
            b. Adjust learning rate
    """
    print_and_log(args, "Start Cosine Embedding training")
    loss_func = nn.CosineEmbeddingLoss()
    model.grad_change(config, use_config=False)
    lr = 1e-5

    optimizer = AdamW(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, threshold=0.001, threshold_mode='abs', patience=5)
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        num_batches = 0

        for batch in tqdm(train_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE):
            batch = {k: v.to(device) for k, v in batch.items() if k not in ['sent_ids', 'sents']}
            optimizer.zero_grad(set_to_none=True)
            embed_1, embed_2 = predict_func(batch['token_ids_1'], batch['attention_mask_1'], batch['token_ids_2'],
                                            batch['attention_mask_2'])
            labels = torch.zeros_like(batch['labels'])
            labels[batch['labels'] >= 3] = 1
            labels[batch['labels'] <= 2] = -1
            loss = loss_func(embed_1, embed_2, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            num_batches += 1

        train_loss /= num_batches
        scheduler.step(train_loss)
        print_and_log(args,
                      f"Epoch {epoch}:: lr :: {scheduler.get_last_lr()[-1] :.6f} train loss :: {train_loss :.5f}")
    model.grad_change(config, use_config=True)


def train(num_of_inputs, args, config, device, model, train_dataloader, dev_dataloader, predict_func, loss_func,
          eval_func):
    """
        Train the model for a specific task.

        Inputs:
        - num_of_inputs: Number of input sentences (1 for sentiment, 2 for paraphrase and similarity)
        - args: Command line arguments
        - config: Model configuration
        - device: Device to run the model on
        - model: The MultitaskBERT model
        - train_dataloader: DataLoader for training data
        - dev_dataloader: DataLoader for development data
        - predict_func: Task-specific prediction function
        - loss_func: Task-specific loss function
        - eval_func: Task-specific evaluation function

        This function fine-tunes the model for a specific task. The flow is:
        1. Set up optimizer and scheduler
        2. For each epoch:
            a. For each batch:
                - Make predictions
                - Compute loss
                - Backpropagate and update model
            b. Evaluate on dev set
            c. Save best model
            d. Adjust learning rate
    """
    lr = args.lr
    optimizer = AdamW(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, threshold=0.001, threshold_mode='abs', patience=5)

    best_dev_score = 0
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        num_batches = 0

        for batch in tqdm(train_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE):
            batch = {k: v.to(device) for k, v in batch.items() if k not in ['sent_ids', 'sents']}
            optimizer.zero_grad(set_to_none=True)

            if num_of_inputs == 2:
                logits = predict_func(batch['token_ids_1'], batch['attention_mask_1'], batch['token_ids_2'],
                                      batch['attention_mask_2'])
                loss = loss_func(logits, batch['labels'].view(args.batch_size, -1).float()) / args.batch_size
            else:
                logits = predict_func(batch["token_ids"], batch["attention_mask"])
                loss = loss_func(logits, batch['labels'].view(-1), reduction='sum') / args.batch_size

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            num_batches += 1

        train_loss /= num_batches
        dev_score = eval_func(dev_dataloader, model, device)

        if dev_score > best_dev_score:
            best_dev_score = dev_score
            save_model(model, optimizer, args, config, args.filepath)
        scheduler.step(train_loss)
        print_and_log(args,
                      f"Epoch {epoch}:: lr :: {scheduler.get_last_lr()[-1] :.6f} train loss :: {train_loss :.5f}, dev "
                      f"acc :: {dev_score :.3f}")


def test_multitask(args):
    """
    Test the model on all three tasks and save predictions.

    Input:
    - args: Command line arguments

    This function performs the following steps:
    1. Load the trained model
    2. Create dataloaders for dev and test sets
    3. Evaluate the model on dev sets for all tasks
    4. Generate predictions for test sets for all tasks
    5. Save the results

    The function uses torch.no_grad() to disable gradient calculation during testing,
    which reduces memory usage and speeds up computation.
    """
    with torch.no_grad():
        device = torch.device('mps') if args.use_gpu else torch.device('cpu')
        saved = torch.load(args.filepath)
        config = saved['model_config']

        model = MultitaskBERT(config)
        model.load_state_dict(saved['model'])
        model = model.to(device)
        print(f"Loaded model to test from {args.filepath}")

        sst_dev_dataloader, para_dev_dataloader, sts_dev_data, sts_dev_dataloader, num_labels = create_dev_dataloaders(
            args)
        sst_test_dataloader, para_test_dataloader, sts_test_dataloader = create_test_dataloaders(args)

        dev_sentiment_accuracy, dev_sst_y_pred, dev_sst_sent_ids, \
            dev_paraphrase_accuracy, dev_para_y_pred, dev_para_sent_ids, \
            dev_sts_corr, dev_sts_y_pred, dev_sts_sent_ids = model_eval_multitask(sst_dev_dataloader,
                                                                                  para_dev_dataloader,
                                                                                  sts_dev_dataloader, model, device)

        test_sst_y_pred, \
            test_sst_sent_ids, test_para_y_pred, test_para_sent_ids, test_sts_y_pred, test_sts_sent_ids = \
            model_eval_test_multitask(sst_test_dataloader,
                                      para_test_dataloader,
                                      sts_test_dataloader, model, device)

        with open(args.sst_dev_out, "w+") as f:
            print(f"dev sentiment acc :: {dev_sentiment_accuracy :.3f}")
            f.write(f"id \t Predicted_Sentiment \n")
            for p, s in zip(dev_sst_sent_ids, dev_sst_y_pred):
                f.write(f"{p} , {s} \n")

        with open(args.sst_test_out, "w+") as f:
            f.write(f"id \t Predicted_Sentiment \n")
            for p, s in zip(test_sst_sent_ids, test_sst_y_pred):
                f.write(f"{p} , {s} \n")

        with open(args.para_dev_out, "w+") as f:
            print(f"dev paraphrase acc :: {dev_paraphrase_accuracy :.3f}")
            f.write(f"id \t Predicted_Is_Paraphrase \n")
            for p, s in zip(dev_para_sent_ids, dev_para_y_pred):
                f.write(f"{p} , {s} \n")

        with open(args.para_test_out, "w+") as f:
            f.write(f"id \t Predicted_Is_Paraphrase \n")
            for p, s in zip(test_para_sent_ids, test_para_y_pred):
                f.write(f"{p} , {s} \n")

        with open(args.sts_dev_out, "w+") as f:
            print(f"dev sts corr :: {dev_sts_corr :.3f}")
            f.write(f"id \t Predicted_Similarity \n")
            for p, s in zip(dev_sts_sent_ids, dev_sts_y_pred):
                f.write(f"{p} , {s} \n")

        with open(args.sts_test_out, "w+") as f:
            f.write(f"id \t Predicted_Similarity \n")
            for p, s in zip(test_sts_sent_ids, test_sts_y_pred):
                f.write(f"{p} , {s} \n")


if __name__ == "__main__":
    args = get_args()
    args.filepath = "multitask_classifier/models/" + f'{args.version}--{args.fine_tune_mode}-{args.epochs}-{args.lr}-multitask.pt'
    args.logpath = "multitask_classifier/logs/" + f'{args.version}--{args.fine_tune_mode}-{args.epochs}-{args.lr}-multitask.txt'
    seed_everything(args.seed)
    train_multitask(args)
    test_multitask(args)
