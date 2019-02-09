{
  "dataset_reader": {
    "type": "se_snli",
    "token_indexers": {
      "tokens": {
        "type": "single_id",
        "lowercase_tokens": true
      }
    },
    "tokenizer": {
      //"end_tokens": ["@@NULL@@"]
    }
  },
  "train_data_path":
  "snli_1.0/snli_1.0_train.jsonl",
  "validation_data_path":
  "snli_1.0/snli_1.0_dev.jsonl",
  "model": {
    "type": "syntactic_entailment",
    "text_field_embedder": {
      "tokens": {
        "type": "embedding",
        "projection_dim": 200,
        "pretrained_file": "glove/glove.6B.300d.txt",
        "embedding_dim": 300,
        "trainable": false
      }
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
      "input_dim": 1400,
      "num_layers": 2,
      "hidden_dims": [200, 3],
      "activations": ["relu", "linear"],
      "dropout": [0.2, 0.0]
    },
    "initializer": [
      [".*linear_layers.*weight", {"type": "xavier_normal"}],
      [".*token_embedder_tokens._projection.*weight", {"type": "xavier_normal"}]
    ],
    "parser_model_path": "models/se-constituency-parser-v4-2/model.tar.gz",
    "predictor_name": "syntactic-entailment-constituency-parser"
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
  }
}