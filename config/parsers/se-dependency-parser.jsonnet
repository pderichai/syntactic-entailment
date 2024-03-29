{
    "dataset_reader": {
        "type": "universal_dependencies",
        "use_language_specific_pos": true
    },
    "iterator": {
        "type": "bucket",
        "batch_size": 128,
        "sorting_keys": [
            [
                "words",
                "num_tokens"
            ]
        ]
    },
    "model": {
        "type": "syntactic-entailment-dependency-parser",
        "arc_representation_dim": 500,
        "dropout": 0.3,
        "encoder": {
            "type": "stacked_bidirectional_lstm",
            "hidden_size": 400,
            "input_size": 200,
            "num_layers": 3,
            "recurrent_dropout_probability": 0.3,
            "use_highway": true
        },
        "initializer": [
            [
                ".*feedforward.*weight",
                {
                    "type": "xavier_uniform"
                }
            ],
            [
                ".*feedforward.*bias",
                {
                    "type": "zero"
                }
            ],
            [
                ".*tag_bilinear.*weight",
                {
                    "type": "xavier_uniform"
                }
            ],
            [
                ".*tag_bilinear.*bias",
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
            ]
        ],
        "input_dropout": 0.3,
        "pos_tag_embedding": {
            "embedding_dim": 100,
            "sparse": true,
            "vocab_namespace": "pos"
        },
        "tag_representation_dim": 100,
        "text_field_embedder": {
            "tokens": {
                "type": "embedding",
                "embedding_dim": 100,
                "pretrained_file": "/glove/glove.6B.100d.txt.gz",
                "sparse": true,
                "trainable": true
            }
        },
        "use_mst_decoding_for_validation": true
    },
    "train_data_path": "/ptb/ptb3.0-stanford.gold.cpos.train.conll",
    "validation_data_path": "/ptb/ptb3.0-stanford.gold.cpos.dev.conll",
    "trainer": {
        "cuda_device": 0,
        "grad_norm": 5,
        "num_epochs": 80,
        "optimizer": {
            "type": "dense_sparse_adam",
            "betas": [
                0.9,
                0.9
            ]
        },
        "patience": 50,
        "validation_metric": "+LAS"
    }
}
