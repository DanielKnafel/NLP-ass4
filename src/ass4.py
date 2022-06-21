"""!pip install datasets transformers[sentencepiece] evaluate"""
import argparse

import evaluate
import numpy as np
from torch import nn
from transformers import Trainer
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding
from transformers import TrainingArguments, BertForNextSentencePrediction




def compute_metrics(eval_preds):
    recall_metric = evaluate.load('recall')
    precision_metric = evaluate.load('precision')
    f1_metric = evaluate.load('f1')
    acc_metric = evaluate.load('accuracy')
    results = {}
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    results.update(recall_metric.compute(predictions=predictions, references=labels))
    results.update(precision_metric.compute(predictions=predictions, references=labels))
    results.update(f1_metric.compute(predictions=predictions, references=labels))
    results.update(acc_metric.compute(predictions=predictions, references=labels))
    return results


class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels.txt")
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")
        # compute custom loss (suppose one has 3 labels.txt with different weights)
        loss_fct = nn.CrossEntropyLoss().to('cuda')
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss




def full_flow(checkpoint,data_files):

    # name = checkpoint + args['data']

    data_collator, tokenized_datasets, tokenizer = tokenize_data(checkpoint, data_files)

    training_args = TrainingArguments("test-trainer", do_train=True, do_eval=True, num_train_epochs=12,
                                      evaluation_strategy='epoch', learning_rate=3e-5,
                                      seed=1,
                                      )
    model = BertForNextSentencePrediction.from_pretrained(checkpoint, num_labels=2)


    predictions = train_and_predict(data_collator, model, tokenized_datasets, tokenizer, training_args)
    return predictions


def train_and_predict(data_collator, model, tokenized_datasets, tokenizer, training_args):
    trainer = CustomTrainer(
        model,
        training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    trainer.save_model(f'model')
    predictions = trainer.predict(tokenized_datasets['validation'])
    return predictions


def tokenize_data(checkpoint, data_files):
    def tokenize_function(example):
        return tokenizer(example["sent1"], example['sent2'], truncation=True)

    raw_datasets = load_dataset("json", data_files=data_files)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
    tokenized_datasets = tokenized_datasets.remove_columns(['idx', 'sent1', 'sent2'])
    tokenized_datasets = tokenized_datasets.rename_column('label', 'labels.txt')
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    return data_collator, tokenized_datasets, tokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", type=str, required=True)
    parser.add_argument("--dev_file", type=str, required=True)
    args = parser.parse_args()
    data_files = {'train': args.train_file, 'validation': args.dev_file}
    checkpoint = "bert-large-cased"
    predictions = full_flow(checkpoint, data_files)


if __name__ == '__main__':
    main()