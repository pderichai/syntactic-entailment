{
  "dataset_reader": {
    "type": "se-bert-snli",
    "pretrained_bert_model_file": "bert-base-uncased",
    "token_indexers": {
      "se-tokens": {
        "namespace": "se-tokens",
        "type": "single_id",
        "lowercase_tokens": true
      },
      "tokens": {
        "type": "single_id"
      }
    },
    "tokenizer": {
      //"end_tokens": ["@@NULL@@"],
      "word_splitter": {
        "type": "spacy",
        "pos_tags": true
      }
    }
  },
  "train_data_path": "SciTailV1.1/snli_format/scitail_1.0_train.txt",
  "validation_data_path": "SciTailV1.1/snli_format/scitail_1.0_dev.txt",
  "model": {
    "type": "se-bert-nli",
    "pretrained_bert_model_file": "bert-base-uncased",
  },
  "iterator": {
    "type": "basic",
    "batch_size": 32
  },
  "trainer": {
    "num_epochs": 3,
    "cuda_device": 0,
    "validation_metric": "+accuracy",
    "optimizer": {
      "type": "adam",
      "lr": 0.00005
    }
  }
}
