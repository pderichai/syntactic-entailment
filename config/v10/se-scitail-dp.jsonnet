{
  "dataset_reader": {
    "type": "se-bert-snli",
    "pretrained_bert_model_file": "pretrained-models/mt-dnn-base",
    "token_indexers": {
      "tokens": {
        "type": "single_id"
      }
    },
    "tokenizer": {
      "word_splitter": {
        "type": "spacy",
        "pos_tags": true
      }
    }
  },
  "train_data_path": "SciTailV1.1/snli_format/scitail_1.0_train.txt",
  "validation_data_path": "SciTailV1.1/snli_format/scitail_1.0_dev.txt",
  "model": {
    "type": "se-bert",
    "parser_model_path": "pretrained-models/se-dependency-parser-v1.tar.gz",
    "parser_hidden_size": 800,
    "parser_cuda_device": 1,
    "freeze_parser": true,
    "pretrained_bert_model_file": "pretrained-models/mt-dnn-base",
    "num_labels": 2
  },
  "iterator": {
    "type": "basic",
    "batch_size": 32
  },
  "trainer": {
    "num_epochs": 5,
    "cuda_device": 1,
    "validation_metric": "+accuracy",
    "optimizer": {
      "type": "bert_adamax",
      "lr": 0.00005,
      "warmup": 0.1,
      "t_total": 10000,
      "schedule": "warmup_linear"
    }
  },
  "vocabulary": {
    "type": "se-vocabulary",
    "parser_vocab": "pretrained-models/se-dependency-parser-v1-vocabulary/tokens.txt",
    "pos_vocab": "pretrained-models/se-dependency-parser-v1-vocabulary/pos.txt"
  }
}
