'''
Multitask BERT class, starter training code, evaluation, and test code.

Of note are:
* class MultitaskBERT: Your implementation of multitask BERT.
* function train_multitask: Training procedure for MultitaskBERT. Starter code
    copies training procedure from `classifier.py` (single-task SST).
* function test_multitask: Test procedure for MultitaskBERT. This function generates
    the required files for submission.

Running `python multitask_classifier.py` trains and tests your MultitaskBERT and
writes all required submission files.
'''

import argparse
import random
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from bert import BertModel
from datasets import (
    SentenceClassificationDataset,
    SentenceClassificationTestDataset,
    SentencePairDataset,
    SentencePairTestDataset,
    load_multitask_data
)
from evaluation import model_eval_sst, model_eval_para, model_eval_sts, model_eval_multitask, model_eval_test_multitask
from optimizer import AdamW

TQDM_DISABLE = False


# Fix the random seed.
def seed_everything(seed=11711):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


BERT_HIDDEN_SIZE = 768
N_SENTIMENT_CLASSES = 5


class MultitaskBERT(nn.Module):
    '''
    This module should use BERT for 3 tasks:

    - Sentiment classification (predict_sentiment)
    - Paraphrase detection (predict_paraphrase)
    - Semantic Textual Similarity (predict_similarity)
    '''

    def __init__(self, config):
        super(MultitaskBERT, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        # last-linear-layer mode does not require updating BERT paramters.
        assert config.fine_tune_mode in ["last-linear-layer", "full-model"]
        for param in self.bert.parameters():
            if config.fine_tune_mode == 'last-linear-layer':
                param.requires_grad = False
            elif config.fine_tune_mode == 'full-model':
                param.requires_grad = True
        # My code:
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.sentiment_last_layer = nn.Linear(config.hidden_size, N_SENTIMENT_CLASSES)
        self.paraphrase_last_layer = nn.Linear(2 * config.hidden_size, 1)
        self.similarity_last_layer = nn.Linear(2 * config.hidden_size, 1)

        # TODO: check size for data compression
        # self.shrink = nn.Linear(config.hidden_size, int(config.hidden_size/2))
        # self.expand = nn.Linear((config.hidden_size/2), config.hidden_size)
        # TODO: try 1D conv
        # self.conv1d_paraphrase = nn.Conv1d(in_channels=2, out_channels=64, kernel_size=3)
        # self.conv1d_similarity = nn.Conv1d(in_channels=2, out_channels=64, kernel_size=3)
        # self.paraphrase_last_layer = nn.Linear(64, 1)
        # self.similarity_last_layer = nn.Linear(64, 1)

    def forward(self, input_ids, attention_mask):
        'Takes a batch of sentences and produces embeddings for them.'
        # The final BERT embedding is the hidden state of [CLS] token (the first token)
        # Here, you can start by just returning the embeddings straight from BERT.
        # When thinking of improvements, you can later try modifying this
        # (e.g., by adding other layers).
        # TODO: implement the shrink/expand
        bert_output = self.bert(input_ids, attention_mask)['pooler_output']
        return bert_output

    def predict_sentiment(self, input_ids, attention_mask):
        '''Given a batch of sentences, outputs logits for classifying sentiment.
        There are 5 sentiment classes:
        (0 - negative, 1- somewhat negative, 2- neutral, 3- somewhat positive, 4- positive)
        Thus, your output should contain 5 logits for each sentence.
        '''
        bert_out = self.forward(input_ids, attention_mask)
        return self.dropout(self.sentiment_last_layer(bert_out))

    def predict_paraphrase(self,
                           input_ids_1, attention_mask_1,
                           input_ids_2, attention_mask_2):
        '''Given a batch of pairs of sentences, outputs a single logit for predicting whether they are paraphrases.
        Note that your output should be unnormalized (a logit); it will be passed to the sigmoid function
        during evaluation.
        '''
        bert_out_1 = self.forward(input_ids_1, attention_mask_1)
        bert_out_2 = self.forward(input_ids_2, attention_mask_2)
        concat_embbed = torch.cat((bert_out_1, bert_out_2), dim=1)
        return self.dropout(self.paraphrase_last_layer(concat_embbed))

        # TODO: try 1D conv
        # Stack embeddings: (batch_size, hidden_size, 2)
        # stacked_embed = torch.stack((bert_out_1, bert_out_2), dim=2)
        # Apply 1D convolution
        # conv_out = F.relu(self.conv1d_paraphrase(stacked_embed))
        # Global max pooling
        # pooled = F.max_pool1d(conv_out, conv_out.size(2)).squeeze(2)
        # Final classification
        # return self.dropout(self.paraphrase_last_layer(pooled))

    def predict_similarity(self,
                           input_ids_1, attention_mask_1,
                           input_ids_2, attention_mask_2):
        '''Given a batch of pairs of sentences, outputs a single logit corresponding to how similar they are.
        Note that your output should be unnormalized (a logit).
        '''
        bert_out_1 = self.forward(input_ids_1, attention_mask_1)
        bert_out_2 = self.forward(input_ids_2, attention_mask_2)
        concat_embbed = torch.cat((bert_out_1, bert_out_2), dim=1)
        return self.dropout(self.similarity_last_layer(concat_embbed))

        # TODO: try 1D conv
        # Stack embeddings: (batch_size, hidden_size, 2)
        # stacked_embed = torch.stack((bert_out_1, bert_out_2), dim=2)
        # Apply 1D convolution
        # conv_out = F.relu(self.conv1d_similarity(stacked_embed))
        # Global max pooling
        # pooled = F.max_pool1d(conv_out, conv_out.size(2)).squeeze(2)
        # Final classification
        # return self.dropout(self.similarity_last_layer(pooled))


def save_model(model, optimizer, args, config, filepath):
    save_info = {
        'model': model.state_dict(),
        'optim': optimizer.state_dict(),
        'args': args,
        'model_config': config,
        'system_rng': random.getstate(),
        'numpy_rng': np.random.get_state(),
        'torch_rng': torch.random.get_rng_state(),
    }

    torch.save(save_info, filepath)
    print(f"save the model to {filepath}")


def create_train_dataloaders(args):
    sst_train_data, num_labels, para_train_data, sts_train_data = load_multitask_data(args.sst_train, args.para_train,
                                                                                      args.sts_train, split='train')

    sst_train_data = SentenceClassificationDataset(sst_train_data, args)
    sst_train_dataloader = DataLoader(sst_train_data, shuffle=True, batch_size=args.batch_size,
                                      collate_fn=sst_train_data.collate_fn)

    para_train_data = SentencePairDataset(para_train_data, args)
    para_train_dataloader = DataLoader(para_train_data, shuffle=False, batch_size=args.batch_size,
                                       collate_fn=para_train_data.collate_fn)

    sts_train_data = SentencePairDataset(sts_train_data, args, isRegression=True)
    sts_train_dataloader = DataLoader(sts_train_data, shuffle=False, batch_size=args.batch_size,
                                      collate_fn=sts_train_data.collate_fn)

    return sst_train_dataloader, para_train_dataloader, sts_train_dataloader


def create_dev_dataloaders(args):
    sst_dev_data, num_labels, para_dev_data, sts_dev_data = \
        load_multitask_data(args.sst_dev, args.para_dev, args.sts_dev, split='dev')

    sst_dev_data = SentenceClassificationDataset(sst_dev_data, args)
    sst_dev_dataloader = DataLoader(sst_dev_data, shuffle=False, batch_size=args.batch_size,
                                    collate_fn=sst_dev_data.collate_fn)

    para_dev_data = SentencePairDataset(para_dev_data, args)
    para_dev_dataloader = DataLoader(para_dev_data, shuffle=False, batch_size=args.batch_size,
                                     collate_fn=para_dev_data.collate_fn)

    sts_dev_data = SentencePairDataset(sts_dev_data, args, isRegression=True)
    sts_dev_dataloader = DataLoader(sts_dev_data, shuffle=False, batch_size=args.batch_size,
                                    collate_fn=sts_dev_data.collate_fn)
    return sst_dev_dataloader, para_dev_dataloader, sts_dev_data, sts_dev_dataloader, num_labels


def create_test_dataloaders(args):
    sst_test_data, num_labels, para_test_data, sts_test_data = \
        load_multitask_data(args.sst_test, args.para_test, args.sts_test, split='test')

    sst_test_data = SentenceClassificationTestDataset(sst_test_data, args)
    sst_test_dataloader = DataLoader(sst_test_data, shuffle=True, batch_size=args.batch_size,
                                     collate_fn=sst_test_data.collate_fn)

    para_test_data = SentencePairTestDataset(para_test_data, args)
    para_test_dataloader = DataLoader(para_test_data, shuffle=True, batch_size=args.batch_size,
                                      collate_fn=para_test_data.collate_fn)

    sts_test_data = SentencePairTestDataset(sts_test_data, args)
    sts_test_dataloader = DataLoader(sts_test_data, shuffle=True, batch_size=args.batch_size,
                                     collate_fn=sts_test_data.collate_fn)

    return sst_test_dataloader, para_test_dataloader, sts_test_dataloader


def train_multitask(args):
    '''Train MultitaskBERT.

    Currently only trains on SST dataset. The way you incorporate training examples
    from other datasets into the training procedure is up to you. To begin, take a
    look at test_multitask below to see how you can use the custom torch `Dataset`s
    in datasets.py to load in examples from the Quora and SemEval datasets.
    '''
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
    for data_set_name, train_dataloader, dev_dataloader, predict_func, loss_func, eval_func in [
        ("para", para_train_dataloader, para_dev_dataloader, model.predict_paraphrase,
         F.binary_cross_entropy_with_logits, model_eval_para),
        ("sst", sst_train_dataloader, sst_dev_dataloader, model.predict_sentiment, F.cross_entropy, model_eval_sst),
        ("sts", sts_train_dataloader, sts_dev_dataloader, model.predict_similarity, F.mse_loss, model_eval_sts)]:
        log_message = "Start training for {} dataset".format(data_set_name)
        print(log_message)
        with open(args.logpath, "a") as f:
            f.write(log_message + "\n")
        print(args, config, device, model, train_dataloader, dev_dataloader, predict_func, loss_func, eval_func)
        train(args, config, device, model, train_dataloader, dev_dataloader, predict_func, loss_func, eval_func)


def train(args, config, device, model, train_dataloader, dev_dataloader, predict_func, loss_func, eval_func):
    lr = args.lr
    optimizer = AdamW(model.parameters(), lr=lr)
    best_dev_score = 0

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        num_batches = 0
        for batch in tqdm(train_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE):
            b_ids, b_mask, b_labels = (batch['token_ids'],
                                       batch['attention_mask'], batch['labels'])

            b_ids = b_ids.to(device)
            b_mask = b_mask.to(device)
            b_labels = b_labels.to(device)

            optimizer.zero_grad()
            logits = predict_func(b_ids, b_mask)
            loss = loss_func(logits, b_labels.view(-1), reduction='sum') / args.batch_size

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            num_batches += 1

        train_loss = train_loss / num_batches

        train_score = eval_func(train_dataloader, model, device)
        dev_score = eval_func(dev_dataloader, model, device)

        if dev_score > best_dev_score:
            best_dev_score = dev_score
            save_model(model, optimizer, args, config, args.filepath)

        log_message = f"Epoch {epoch}: train loss :: {train_loss :.3f}, train acc :: {train_score :.3f}, dev acc :: {dev_score :.3f}"
        print(log_message)

        with open(args.logpath, "a") as f:
            f.write(log_message + "\n")


def test_multitask(args):
    '''Test and save predictions on the dev and test sets of all three tasks.'''
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
            f.write(f"id \t Predicted_Similiary \n")
            for p, s in zip(dev_sts_sent_ids, dev_sts_y_pred):
                f.write(f"{p} , {s} \n")

        with open(args.sts_test_out, "w+") as f:
            f.write(f"id \t Predicted_Similiary \n")
            for p, s in zip(test_sts_sent_ids, test_sts_y_pred):
                f.write(f"{p} , {s} \n")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sst_train", type=str, default="data/ids-sst-train.csv")
    parser.add_argument("--sst_dev", type=str, default="data/ids-sst-dev.csv")
    parser.add_argument("--sst_test", type=str, default="data/ids-sst-test-student.csv")

    parser.add_argument("--para_train", type=str, default="data/quora-train.csv")
    parser.add_argument("--para_dev", type=str, default="data/quora-dev.csv")
    parser.add_argument("--para_test", type=str, default="data/quora-test-student.csv")

    parser.add_argument("--sts_train", type=str, default="data/sts-train.csv")
    parser.add_argument("--sts_dev", type=str, default="data/sts-dev.csv")
    parser.add_argument("--sts_test", type=str, default="data/sts-test-student.csv")

    parser.add_argument("--seed", type=int, default=11711)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--fine-tune-mode", type=str,
                        help='last-linear-layer: the BERT parameters are frozen and the task specific head parameters are updated; full-model: BERT parameters are updated as well',
                        choices=('last-linear-layer', 'full-model'), default="last-linear-layer")
    parser.add_argument("--use_gpu", default='True')

    parser.add_argument("--sst_dev_out", type=str, default="multitask_classifier/predictions/sst-dev-output.csv")
    parser.add_argument("--sst_test_out", type=str, default="multitask_classifier/predictions/sst-test-output.csv")

    parser.add_argument("--para_dev_out", type=str, default="multitask_classifier/predictions/para-dev-output.csv")
    parser.add_argument("--para_test_out", type=str, default="multitask_classifier/predictions/para-test-output.csv")

    parser.add_argument("--sts_dev_out", type=str, default="multitask_classifier/predictions/sts-dev-output.csv")
    parser.add_argument("--sts_test_out", type=str, default="multitask_classifier/predictions/sts-test-output.csv")

    parser.add_argument("--batch_size", help='sst: 64, cfimdb: 8 can fit a 12GB GPU', type=int, default=8)
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.3)
    parser.add_argument("--lr", type=float, help="learning rate", default=1e-5)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    args.filepath = "multitask_classifier/models/" + f'{args.fine_tune_mode}-{args.epochs}-{args.lr}-multitask.pt'
    args.logpath = "multitask_classifier/logs/" + f'{args.fine_tune_mode}-{args.epochs}-{args.lr}-multitask.txt'
    seed_everything(args.seed)
    train_multitask(args)
    test_multitask(args)
