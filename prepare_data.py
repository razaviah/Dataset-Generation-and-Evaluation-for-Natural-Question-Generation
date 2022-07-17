import logging
import os
from dataclasses import dataclass, field
from typing import Optional

import torch
from nlp import load_dataset
from transformers import T5Tokenizer, BartTokenizer, HfArgumentParser

logger = logging.getLogger(__name__)


@dataclass
class DataTrainingArguments:
    model_type: Optional[str] = field(
        default='bart',
        metadata={"help": "One of 't5', 'bart'"},
    )
    max_source_length: Optional[int] = field(
        default=512,
        metadata={"help": "Max input length for the source text"},
    )
    max_target_length: Optional[int] = field(
        default=512,
        metadata={"help": "Max input length for the target text"},
    )


class DataProcessor:
    def __init__(self, tokenizer, model_type="bart", max_source_length=512, max_target_length=512):
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.model_type = model_type

        if model_type == "t5":
            self.sep_token = "<sep>"
        elif model_type == "bart":
            self.sep_token = "<sep>"
        else:
            self.sep_token = "[SEP]"

    def process(self, dataset):
        if self.model_type == "t5":
            dataset = dataset.map(self._add_eos_examples)

        dataset = dataset.map(self._add_special_tokens)
        dataset = dataset.map(self._convert_to_features, batched=True)

        return dataset

    def _add_eos_examples(self, example):
        example['source_text'] = example['source_text'] + " </s>"
        example['target_text'] = example['target_text'] + " </s>"
        return example

    def _add_special_tokens(self, example):
        example['target_text'] = example['target_text'].replace("{sep_token}", self.sep_token)
        return example

    def _convert_to_features(self, example_batch):
        source_encoding = self.tokenizer.batch_encode_plus(
            example_batch['source_text'],
            max_length=self.max_source_length,
            padding='max_length',
            pad_to_max_length=True,
            truncation=True,
        )
        target_encoding = self.tokenizer.batch_encode_plus(
            example_batch['target_text'],
            max_length=self.max_target_length,
            padding='max_length',
            pad_to_max_length=True,
            truncation=True,
        )

        encodings = {
            'source_ids': source_encoding['input_ids'],
            'target_ids': target_encoding['input_ids'],
            'attention_mask': source_encoding['attention_mask'],
        }

        return encodings


def main():
    parser = HfArgumentParser((DataTrainingArguments,))
    data_args = parser.parse_args_into_dataclasses()[0]

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO
    )

    if data_args.model_type == 't5':
        tokenizer = T5Tokenizer.from_pretrained("t5-base")
    else:
        tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")

    #tokenizer.add_tokens(['<sep>'])
    tokenizer.add_tokens(
        ['<sep>', '$R-ARGM-ADV', '$R-ARGM-CAU', '$C-ARG2', '$ARGM-LVB', '$ARGM-COM', '$ARGA', '$ARGM-ADJ', '$ARGM-PNC',
         '$ARG0', '$ARGM-MNR', '$C-ARGM-LOC', '$C-ARGM-MNR', '$C-ARG4', '$ARGM-DIR', '$ARGM-EXT', '$R-ARG1',
         '$R-ARGM-EXT', '$ARGM-LOC', '$R-ARGM-MNR', '$ARGM-PRR', '$C-ARG1', '$ARGM-ADV', '$ARGM-MOD', '$ARGM-REC',
         '$R-ARG2', '$C-ARGM-EXT', '$ARGM-PRP', '$ARGM-DIS', '$ARG3', '$ARGM-TMP', '$R-ARG3', '$ARGM-GOL', '$R-ARG0',
         '$C-ARGM-ADV', '$ARG1', '$ARGM-CAU', '$C-ARG0', '$V', '$ARG4', '$R-ARGM-TMP', '$ARGM-PRD', '$ARG5', '$ARG2',
         '$C-ARGM-TMP', '$ARGM-NEG', '$R-ARGM-DIR', '$R-ARGM-LOC', '$R-ARGM-GOL'])

    train_dataset = load_dataset('csv', data_files=['data/single/train.csv'], delimiter='\t')['train'] ######################
    valid_dataset = load_dataset('csv', data_files=['data/single/dev.csv'], delimiter='\t')['train']   ######################
    test_dataset = load_dataset('csv', data_files=['data/single/test.csv'], delimiter='\t')['train']   ######################

    processor = DataProcessor(
        tokenizer,
        model_type=data_args.model_type,
        max_source_length=data_args.max_source_length,
        max_target_length=data_args.max_target_length
    )

    train_dataset = processor.process(train_dataset)
    valid_dataset = processor.process(valid_dataset)
    test_dataset = processor.process(test_dataset)

    columns = ["source_ids", "target_ids", "attention_mask"]
    train_dataset.set_format(type='torch', columns=columns)
    valid_dataset.set_format(type='torch', columns=columns)
    test_dataset.set_format(type='torch', columns=columns)

    train_file_name = f"train_data_{data_args.model_type}.pt"
    train_path = os.path.join("data", train_file_name) ##################
    valid_file_name = f"valid_data_{data_args.model_type}.pt"
    valid_path = os.path.join("data", valid_file_name) ##################
    test_file_name = f"test_data_{data_args.model_type}.pt"
    test_path = os.path.join("data", test_file_name)   ##################

    torch.save(train_dataset, train_path)
    logger.info(f"saved train dataset at {train_path}")
    torch.save(valid_dataset, valid_path)
    logger.info(f"saved validation dataset at {valid_path}")
    torch.save(test_dataset, test_path)
    logger.info(f"saved validation dataset at {test_path}")

    tokenizer_path = f"{data_args.model_type}_tokenizer"
    if not os.path.exists(tokenizer_path):
        os.mkdir(tokenizer_path)
    tokenizer.save_pretrained(tokenizer_path)
    logger.info(f"saved tokenizer at {tokenizer_path}")


if __name__ == "__main__":
    main()

