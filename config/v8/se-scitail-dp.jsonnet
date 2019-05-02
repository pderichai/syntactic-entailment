{
  "dataset_reader": {
    "type": "se-snli-v2",
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
    "type": "syntactic-entailment-v8",
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
      "input_dim": 800,
      "num_layers": 2,
      "hidden_dims": [400, 200],
      "activations": "relu",
      "dropout": 0.2
    },
    "similarity_function": {"type": "dot_product"},
    "compare_feedforward": {
      "input_dim": 1600,
      "num_layers": 2,
      "hidden_dims": [400, 200],
      "activations": "relu",
      "dropout": 0.2
    },
    "aggregate_feedforward": {
      "input_dim": 400,
      "num_layers": 2,
      "hidden_dims": [200, 2],
      "activations": ["relu", "linear"],
      "dropout": [0.2, 0.0]
    },
    "initializer": [
      [".*linear_layers.*weight", {"type": "xavier_normal"}],
      [".*token_embedder_se-tokens._projection.*weight", {"type": "xavier_normal"}],
      [".*_parser.*", "prevent"]
    ],
    "parser_model_path": "pretrained-models/se-dependency-parser-v1.tar.gz",
    "parser_cuda_device": 0,
    "freeze_parser": true
  },
  "iterator": {
    "type": "bucket",
    "sorting_keys": [["premise", "num_tokens"], ["hypothesis", "num_tokens"]],
    "batch_size": 64
  },
  "trainer": {
    "num_epochs": 140,
    "patience": 20,
    "cuda_device": 0,
    "grad_clipping": 5.0,
    "validation_metric": "+accuracy",
    "optimizer": {
      "type": "adam"
    }
  },
  "vocabulary": {
    "type": "se-vocabulary",
    "parser_vocab": "pretrained-models/se-dependency-parser-v1-vocabulary/tokens.txt",
    "pos_vocab": "pretrained-models/se-dependency-parser-v1-vocabulary/pos.txt"
  }
}
