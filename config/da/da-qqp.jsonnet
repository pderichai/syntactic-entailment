{
    "dataset_reader": {
        "type": "se-qqp",
        "token_indexers": {
            "tokens": {
                "type": "single_id",
                "lowercase_tokens": true
            }
        },
        "tokenizer": {
            "word_splitter": {
                "type": "spacy",
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
        "type": "decomposable_attention",
        "aggregate_feedforward": {
            "activations": [
                "relu",
                "linear"
            ],
            "dropout": [
                0.535109429080117,
                0
            ],
            "hidden_dims": [
                172,
                2
            ],
            "input_dim": 400,
            "num_layers": 2
        },
        "attend_feedforward": {
            "activations": "relu",
            "dropout": 0.29390420271320783,
            "hidden_dims": [
                295,
                200
            ],
            "input_dim": 200,
            "num_layers": 2
        },
        "compare_feedforward": {
            "activations": "relu",
            "dropout": 0.34408357390601785,
            "hidden_dims": [
                108,
                200
            ],
            "input_dim": 400,
            "num_layers": 2
        },
        "initializer": [
            [
                ".*linear_layers.*weight",
                {
                    "type": "xavier_normal"
                }
            ],
            [
                ".*token_embedder_tokens\\._projection.*weight",
                {
                    "type": "xavier_normal"
                }
            ]
        ],
        "similarity_function": {
            "type": "dot_product"
        },
        "text_field_embedder": {
            "token_embedders": {
                "tokens": {
                    "type": "embedding",
                    "embedding_dim": 300,
                    "pretrained_file": "glove/glove.6B.300d.txt",
                    "projection_dim": 200,
                    "trainable": false
                }
            }
        }
    },
    "train_data_path": "glue_data/QQP/train.tsv",
    "validation_data_path": "glue_data/QQP/dev.tsv",
    "trainer": {
        "cuda_device": 0,
        "grad_clipping": 5,
        "num_epochs": 140,
        "num_serialized_models_to_keep": 0,
        "optimizer": {
            "type": "adam",
            "lr": 0.0003137788286453939
        },
        "patience": 20,
        "validation_metric": "+accuracy"
    }
}
