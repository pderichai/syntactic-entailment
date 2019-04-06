{
  "dataset_reader": {
    "type": "se_snli",
    "token_indexers": {
      "tokens": {
        "type": "single_id",
        "lowercase_tokens": true
      },
      "elmo": {
        "type": "elmo_characters"
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
  "train_data_path":
  "SciTailV1.1/snli_format/scitail_1.0_train.txt",
  "validation_data_path":
  "SciTailV1.1/snli_format/scitail_1.0_dev.txt",
  "model": {
    "type": "syntactic-entailment-v3",
    "text_field_embedder": {
      "tokens": {
        "type": "embedding",
        "projection_dim": 200,
        "pretrained_file": "glove/glove.6B.300d.txt",
        "embedding_dim": 300,
        "trainable": false
      },
      "elmo": {
        "type": "elmo_token_embedder",
        "options_file": "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json",
        "weight_file": "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5",
        "do_layer_norm": false,
        "dropout": 0.5
      }
    },
    "attend_feedforward": {
      "input_dim": 1224,
      "num_layers": 2,
      "hidden_dims": 200,
      "activations": "relu",
      "dropout": 0.2
    },
    "similarity_function": {"type": "dot_product"},
    "compare_feedforward": {
      "input_dim": 2448,
      "num_layers": 2,
      "hidden_dims": 200,
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
      [".*token_embedder_tokens._projection.*weight", {"type": "xavier_normal"}]
    ],
    "parser_model_path": "pretrained-models/se-constituency-parser-v2.tar.gz",
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