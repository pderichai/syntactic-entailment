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
        },
    },
    "train_data_path": "multinli_1.0/multinli_1.0_train.jsonl",
    "validation_data_path": "multinli_1.0/multinli_1.0_dev_matched.jsonl",
    "model": {
        "type": "esim-lf",
        "dropout": 0.5,
        "text_field_embedder": {
            "token_embedders": {
                "se-tokens": {
                    "type": "embedding",
                    "vocab_namespace": "se-tokens",
                    "pretrained_file": "glove/glove.840B.300d.txt",
                    "embedding_dim": 300,
                    "trainable": true
                }
            },
            "allow_unmatched_keys": true
        },
        "encoder": {
            "type": "lstm",
            "input_size": 300,
            "hidden_size": 300,
            "num_layers": 1,
            "bidirectional": true
        },
        "similarity_function": {
            "type": "dot_product"
        },
        "projection_feedforward": {
            "input_dim": 2400,
            "hidden_dims": 300,
            "num_layers": 1,
            "activations": "relu"
        },
        "inference_encoder": {
            "type": "lstm",
            "input_size": 300,
            "hidden_size": 300,
            "num_layers": 1,
            "bidirectional": true
        },
        "output_feedforward": {
            "input_dim": 4000,
            "num_layers": 2,
            "hidden_dims": [2000, 300],
            "activations": "relu",
            "dropout": 0.5
        },
        "output_logit": {
            "input_dim": 300,
            "num_layers": 1,
            "hidden_dims": 3,
            "activations": "linear"
        },
        "parser_model_path": "pretrained-models/biaffine-dependency-parser-ptb-2018.08.23/biaffine-dependency-parser-ptb-2018.08.23.tar.gz",
        "parser_cuda_device": 0,
        "freeze_parser": false,
        "initializer": [
            [".*linear_layers.*weight", {"type": "xavier_uniform"}],
            [".*linear_layers.*bias", {"type": "zero"}],
            [".*weight_ih.*", {"type": "xavier_uniform"}],
            [".*weight_hh.*", {"type": "orthogonal"}],
            [".*bias_ih.*", {"type": "zero"}],
            [".*bias_hh.*", {"type": "lstm_hidden_bias"}],
            [".*_parser.*", "prevent"]
        ]
    },
    "iterator": {
        "type": "bucket",
        "sorting_keys": [["premise", "num_tokens"],
                         ["hypothesis", "num_tokens"]],
        "batch_size": 32
    },
    "trainer": {
        "optimizer": {
            "type": "adam",
            "lr": 0.0004
        },
        "validation_metric": "+accuracy",
        "num_epochs": 75,
        "num_serialized_models_to_keep": 0,
        "grad_norm": 10.0,
        "patience": 5,
        "cuda_device": 0,
        "learning_rate_scheduler": {
            "type": "reduce_on_plateau",
            "factor": 0.5,
            "mode": "max",
            "patience": 0
        }
    },
    "vocabulary": {
      "type": "se-vocabulary",
      "parser_vocab": "pretrained-models/biaffine-dependency-parser-ptb-2018.08.23/vocabulary/tokens.txt",
      "pos_vocab": "pretrained-models/biaffine-dependency-parser-ptb-2018.08.23/vocabulary/pos.txt"
    }
}
