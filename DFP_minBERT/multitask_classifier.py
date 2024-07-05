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
    '''
    This module should use BERT for 3 tasks:

    - Sentiment classification (predict_sentiment)
    - Paraphrase detection (predict_paraphrase)
    - Semantic Textual Similarity (predict_similarity)
    Model Summary:
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
                (out_dense): Linear(in_features=3072, out_features=768, bias=True)
                (out_layer_norm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                (out_dropout): Dropout(p=0.1, inplace=False)
              )
            )

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
        self.r = 32
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.sentiment_last_layer = nn.Linear(config.hidden_size, N_SENTIMENT_CLASSES)
        self.paraphrase_last_layer = nn.Linear(2 * config.hidden_size, 1)
        self.similarity_layer_1 = nn.Linear(config.hidden_size, self.r)
        self.similarity_layer_2 = nn.Linear(config.hidden_size, self.r)
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        self.sigmoid = nn.Sigmoid()

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
        bert_out_1 = self.dropout(self.similarity_layer_1(self.forward(input_ids_1, attention_mask_1)))
        bert_out_2 = self.dropout(self.similarity_layer_2(self.forward(input_ids_2, attention_mask_2)))
        bert_out_1_shaped = bert_out_1.view(bert_out_1.shape[0], 1, bert_out_1.shape[1])
        bert_out_2_shaped = bert_out_2.view(bert_out_1.shape[0], bert_out_1.shape[1], 1)
        out = 5 * (bert_out_1_shaped @ bert_out_2_shaped).reshape(input_ids_1.shape[0], -1)
        # out = 5 * self.sigmoid(self.cos(bert_out_1, bert_out_2).unsqueeze(-1))
        return out

        # TODO: try 1D conv
        # Stack embeddings: (batch_size, hidden_size, 2)
        # stacked_embed = torch.stack((bert_out_1, bert_out_2), dim=2)
        # Apply 1D convolution
        # conv_out = F.relu(self.conv1d_similarity(stacked_embed))
        # Global max pooling
        # pooled = F.max_pool1d(conv_out, conv_out.size(2)).squeeze(2)
        # Final classification
        # return self.dropout(self.similarity_last_layer(pooled))


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
    for data_set_name, inputs, train_dataloader, dev_dataloader, predict_func, loss_func, eval_func in [
        ("sts", 2, sts_train_dataloader, sts_dev_dataloader, model.predict_similarity, F.l1_loss, model_eval_sts),
        ("sst", 1, sst_train_dataloader, sst_dev_dataloader, model.predict_sentiment, F.cross_entropy, model_eval_sst),
        ("para", 2, para_train_dataloader, para_dev_dataloader, model.predict_paraphrase,
         nn.BCEWithLogitsLoss(reduction="sum"), model_eval_para)
    ]:
        print_and_log(args, "Start training for {} dataset".format(data_set_name))
        train(inputs, args, config, device, model, train_dataloader, dev_dataloader, predict_func, loss_func, eval_func)


def train(inputs, args, config, device, model, train_dataloader, dev_dataloader, predict_func, loss_func, eval_func):
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

            if inputs == 2:
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
                      f"Epoch {epoch}:: lr :: {scheduler.get_last_lr()[-1] :.6f} train loss :: {train_loss :.5f}, dev acc :: {dev_score :.3f}")


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


if __name__ == "__main__":
    args = get_args()
    args.filepath = "multitask_classifier/models/" + f'{args.version}--{args.fine_tune_mode}-{args.epochs}-{args.lr}-multitask.pt'
    args.logpath = "multitask_classifier/logs/" + f'{args.version}--{args.fine_tune_mode}-{args.epochs}-{args.lr}-multitask.txt'
    seed_everything(args.seed)
    train_multitask(args)
    test_multitask(args)
