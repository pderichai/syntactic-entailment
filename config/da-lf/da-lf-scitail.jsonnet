{
    "dataset_reader": {
        "type": "se-snli",
        "token_indexers": {
            "se-tokens": {
                "type": "single_id",
                "lowercase_tokens": true,
                "namespace": "se-tokens"
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
    "iterator": {
        "type": "bucket",
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
        ]
    },
    "model": {
        "type": "da-lf",
        "aggregate_feedforward": {
            "activations": [
                "relu",
                "linear"
            ],
            "dropout": [
                0.5301118556970255,
                0
            ],
            "hidden_dims": [
                270,
                2
            ],
            "input_dim": 2000,
            "num_layers": 2
        },
        "attend_feedforward": {
            "activations": "relu",
            "dropout": 0.25969847233598436,
            "hidden_dims": [
                226,
                200
            ],
            "input_dim": 200,
            "num_layers": 2
        },
        "compare_feedforward": {
            "activations": "relu",
            "dropout": 0.4424885293307621,
            "hidden_dims": [
                122,
                200
            ],
            "input_dim": 400,
            "num_layers": 2
        },
        "freeze_parser": true,
        "initializer": [
            [
                ".*linear_layers.*weight",
                {
                    "type": "xavier_normal"
                }
            ],
            [
                ".*token_embedder_se-tokens._projection.*weight",
                {
                    "type": "xavier_normal"
                }
            ],
            [
                ".*_parser.*",
                "prevent"
            ]
        ],
        "parser_cuda_device": 0,
        "parser_model_path": "pretrained-models/biaffine-dependency-parser-ptb-2018.08.23/biaffine-dependency-parser-ptb-2018.08.23.tar.gz",
        "similarity_function": {
            "type": "dot_product"
        },
        "text_field_embedder": {
            "allow_unmatched_keys": true,
            "token_embedders": {
                "se-tokens": {
                    "type": "embedding",
                    "embedding_dim": 300,
                    "pretrained_file": "glove/glove.6B.300d.txt",
                    "projection_dim": 200,
                    "trainable": false,
                    "vocab_namespace": "se-tokens"
                }
            }
        }
    },
    "train_data_path": "SciTailV1.1/snli_format/scitail_1.0_train.txt",
    "validation_data_path": "SciTailV1.1/snli_format/scitail_1.0_dev.txt",
    "trainer": {
        "cuda_device": 0,
        "grad_clipping": 5,
        "num_epochs": 140,
        "num_serialized_models_to_keep": 0,
        "optimizer": {
            "type": "adam",
            "lr": 0.0002936233885589455
        },
        "patience": 10,
        "validation_metric": "+accuracy"
    },
    "vocabulary": {
        "type": "se-vocabulary",
        "parser_vocab": "pretrained-models/biaffine-dependency-parser-ptb-2018.08.23/vocabulary/tokens.txt",
        "pos_vocab": "pretrained-models/biaffine-dependency-parser-ptb-2018.08.23/vocabulary/pos.txt"
    }
}