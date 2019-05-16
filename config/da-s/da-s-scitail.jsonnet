{
    "dataset_reader": {
        "type": "se-snli-v2",
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
        "type": "syntactic-entailment-v1",
        "aggregate_feedforward": {
            "activations": [
                "relu",
                "linear"
            ],
            "dropout": [
                0.4498685529912684,
                0
            ],
            "hidden_dims": [
                155,
                2
            ],
            "input_dim": 2000,
            "num_layers": 2
        },
        "attend_feedforward": {
            "activations": "relu",
            "dropout": 0.4212691529078526,
            "hidden_dims": [
                261,
                200
            ],
            "input_dim": 200,
            "num_layers": 2
        },
        "compare_feedforward": {
            "activations": "relu",
            "dropout": 0.37297283660919067,
            "hidden_dims": [
                397,
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
        "parser_model_path": "pretrained-models/se-dependency-parser-v1.tar.gz",
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
            "lr": 0.0003569843026758792
        },
        "patience": 10,
        "validation_metric": "+accuracy"
    },
    "vocabulary": {
        "type": "se-vocabulary",
        "parser_vocab": "pretrained-models/se-dependency-parser-v1-vocabulary/tokens.txt",
        "pos_vocab": "pretrained-models/se-dependency-parser-v1-vocabulary/pos.txt"
    }
}