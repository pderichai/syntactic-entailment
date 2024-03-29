{
  "dataset_reader": {
    "type": "se-snli",
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
      "word_splitter": {
        "type": "spacy",
        "pos_tags": true
      }
    }
  },
  "train_data_path": "multinli_1.0/multinli_1.0_train.jsonl",
  "validation_data_path": "multinli_1.0/multinli_1.0_dev_matched.jsonl",
  "model": {
    "type": "da-sa",
    "text_field_embedder": {
      "token_embedders": {
        "se-tokens": {
          "type": "embedding",
          "projection_dim": 200,
          "pretrained_file": "glove/glove.6B.300d.txt",
          "embedding_dim": 300,
          "trainable": false,
        }
      },
      "allow_unmatched_keys": true
    },
    "attend_feedforward": {
      "input_dim": 200,
      "num_layers": 2,
      "hidden_dims": 200,
      "activations": "relu",
      "dropout": 0.2
    },
    "similarity_function": {"type": "dot_product"},
    "compare_feedforward": {
      "input_dim": 400,
      "num_layers": 2,
      "hidden_dims": 200,
      "activations": "relu",
      "dropout": 0.2
    },
    "aggregate_feedforward": {
      "input_dim": 400,
      "num_layers": 2,
      "hidden_dims": [200, 3],
      "activations": ["relu", "linear"],
      "dropout": [0.2, 0.0]
    },
    "initializer": [
      [".*linear_layers.*weight", {"type": "xavier_normal"}],
      [".*token_embedder_se-tokens._projection.*weight", {"type": "xavier_normal"}],
      [".*_parser.*", "prevent"]
    ],
    "parser_model_path": "pretrained-models/biaffine-dependency-parser-ptb-2018.08.23/biaffine-dependency-parser-ptb-2018.08.23.tar.gz",
    "parser_cuda_device": 0,
    "freeze_parser": false
  },
  "iterator": {
    "type": "bucket",
    "sorting_keys": [["premise", "num_tokens"], ["hypothesis", "num_tokens"]],
    "batch_size": 64
  },
  "trainer": {
    "num_epochs": 140,
    "num_serialized_models_to_keep": 0,
    "patience": 10,
    "cuda_device": 0,
    "grad_clipping": 5.0,
    "validation_metric": "+accuracy",
    "optimizer": {
      "type": "adam"
    }
  },
  "vocabulary": {
    "type": "se-vocabulary",
    "parser_vocab": "pretrained-models/biaffine-dependency-parser-ptb-2018.08.23/vocabulary/tokens.txt",
    "pos_vocab": "pretrained-models/biaffine-dependency-parser-ptb-2018.08.23/vocabulary/pos.txt"
  }
}
