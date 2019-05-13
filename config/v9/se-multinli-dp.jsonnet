{
  "dataset_reader": {
    "type": "se-bert-snli",
    "pretrained_bert_model_file": "bert-base-uncased",
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
  "train_data_path": "multinli_1.0/multinli_1.0_train.jsonl",
  "validation_data_path": "multinli_1.0/multinli_1.0_dev_matched.jsonl",
  "model": {
    "type": "se-bert",
    "parser_model_path": "pretrained-models/se-dependency-parser-v1.tar.gz",
    "parser_hidden_size": 800,
    "parser_cuda_device": 0,
    "freeze_parser": true,
    "pretrained_bert_model_file": "bert-base-uncased",
    "num_labels": 3
  },
  "iterator": {
    "type": "basic",
    "batch_size": 32
  },
  "trainer": {
    "num_epochs": 5,
    "cuda_device": 0,
    "validation_metric": "+accuracy",
    "grad_clipping": 1.0,
    "optimizer": {
      "type": "adam",
      "lr": 0.00005
    }
  },
  "vocabulary": {
    "type": "se-vocabulary",
    "parser_vocab": "pretrained-models/se-dependency-parser-v1-vocabulary/tokens.txt",
    "pos_vocab": "pretrained-models/se-dependency-parser-v1-vocabulary/pos.txt"
  }
}
