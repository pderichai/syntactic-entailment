{
    "dataset_reader": {
        "token_indexers": {
            "se-tokens": {
                "lowercase_tokens": true,
                "namespace": "se-tokens",
                "type": "single_id"
            },
            "tokens": {
                "type": "single_id"
            }
        },
        "tokenizer": {
            "word_splitter": {
                "pos_tags": true,
                "type": "spacy"
            }
        },
        "type": "se-snli"
    },
    "iterator": {
        "batch_size": 64,
        "sorting_keys": [
            [
                "premise",
                "num_tokens"
            ],
            [
                "hypothesis",
                "num_tokens"
            ]
        ],
        "type": "bucket"
    },
    "model": {
        "dropout": 0.2793419860772328,
        "encoder": {
            "bidirectional": true,
            "hidden_size": 300,
            "input_size": 300,
            "num_layers": 1,
            "type": "lstm"
        },
        "freeze_parser": true,
        "inference_encoder": {
            "bidirectional": true,
            "hidden_size": 300,
            "input_size": 300,
            "num_layers": 1,
            "type": "lstm"
        },
        "initializer": [
            [
                ".*linear_layers.*weight",
                {
                    "type": "xavier_uniform"
                }
            ],
            [
                ".*linear_layers.*bias",
                {
                    "type": "zero"
                }
            ],
            [
                ".*weight_ih.*",
                {
                    "type": "xavier_uniform"
                }
            ],
            [
                ".*weight_hh.*",
                {
                    "type": "orthogonal"
                }
            ],
            [
                ".*bias_ih.*",
                {
                    "type": "zero"
                }
            ],
            [
                ".*bias_hh.*",
                {
                    "type": "lstm_hidden_bias"
                }
            ],
            [
                ".*_parser.*",
                "prevent"
            ]
        ],
        "output_feedforward": {
            "activations": "relu",
            "dropout": 0.3390632597471799,
            "hidden_dims": [
                2000,
                300
            ],
            "input_dim": 4000,
            "num_layers": 2
        },
        "output_logit": {
            "activations": "linear",
            "hidden_dims": 2,
            "input_dim": 300,
            "num_layers": 1
        },
        "parser_cuda_device": 0,
        "parser_model_path": "pretrained-models/biaffine-dependency-parser-ptb-2018.08.23/biaffine-dependency-parser-ptb-2018.08.23.tar.gz",
        "projection_feedforward": {
            "activations": "relu",
            "hidden_dims": 300,
            "input_dim": 2400,
            "num_layers": 1
        },
        "similarity_function": {
            "type": "dot_product"
        },
        "text_field_embedder": {
            "allow_unmatched_keys": true,
            "token_embedders": {
                "se-tokens": {
                    "embedding_dim": 300,
                    "pretrained_file": "glove/glove.840B.300d.txt",
                    "trainable": true,
                    "type": "embedding",
                    "vocab_namespace": "se-tokens"
                }
            }
        },
        "type": "esim-lf"
    },
    "train_data_path": "SciTailV1.1/snli_format/scitail_1.0_train.txt",
    "trainer": {
        "cuda_device": 0,
        "grad_norm": 10,
        "learning_rate_scheduler": {
            "factor": 0.5,
            "mode": "max",
            "patience": 0,
            "type": "reduce_on_plateau"
        },
        "num_epochs": 75,
        "num_serialized_models_to_keep": 0,
        "optimizer": {
            "lr": 0.0006110461682192319,
            "type": "adam"
        },
        "patience": 5,
        "validation_metric": "+accuracy"
    },
    "validation_data_path": "SciTailV1.1/snli_format/scitail_1.0_dev.txt",
    "vocabulary": {
        "parser_vocab": "pretrained-models/biaffine-dependency-parser-ptb-2018.08.23/vocabulary/tokens.txt",
        "pos_vocab": "pretrained-models/biaffine-dependency-parser-ptb-2018.08.23/vocabulary/pos.txt",
        "type": "se-vocabulary"
    }
}
