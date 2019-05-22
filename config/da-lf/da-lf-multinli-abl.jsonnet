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
        "type": "da-lf-noise-abl",
        "aggregate_feedforward": {
            "activations": [
                "relu",
                "linear"
            ],
            "dropout": [
                0.3,
                0
            ],
            "hidden_dims": [
                150,
                3
            ],
            "input_dim": 2000,
            "num_layers": 2
        },
        "attend_feedforward": {
            "activations": "relu",
            "dropout": 0.3,
            "hidden_dims": [
                250,
                200
            ],
            "input_dim": 200,
            "num_layers": 2
        },
        "compare_feedforward": {
            "activations": "relu",
            "dropout": 0.3,
            "hidden_dims": [
                400,
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
        "parser_cuda_device": 1,
        "parser_model_path": "pretrained-models/biaffine-dependency-parser-ptb.tar.gz",
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
    "train_data_path": "multinli_1.0/multinli_1.0_train.jsonl",
    "validation_data_path": "multinli_1.0/multinli_1.0_dev_matched.jsonl",
    "trainer": {
        "cuda_device": 1,
        "grad_clipping": 5,
        "num_epochs": 140,
        "num_serialized_models_to_keep": 0,
        "optimizer": {
            "type": "adam",
            "lr": 0.0003
        },
        "patience": 10,
        "validation_metric": "+accuracy"
    },
    "vocabulary": {
        "type": "se-vocabulary",
        "parser_vocab": "pretrained-models/biaffine-dependency-parser-ptb-vocab/tokens.txt",
        "pos_vocab": "pretrained-models/biaffine-dependency-parser-ptb-vocab/pos.txt"
    }
}
