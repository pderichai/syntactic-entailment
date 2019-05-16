{
  "dataset_reader": {
    "type": "bert-snli",
    "pretrained_bert_model_file": "bert-base-uncased",
  },
  "train_data_path": "snli_1.0/snli_1.0_train.jsonl",
  "validation_data_path": "snli_1.0/snli_1.0_dev.jsonl",
  "model": {
    "type": "bert-nli",
    "pretrained_bert_model_file": "bert-base-uncased",
    "num_labels": 3,
  },
  "iterator": {
    "type": "basic",
    "batch_size": 32
  },
  "trainer": {
    "num_epochs": 4,
    "cuda_device": 0,
    "validation_metric": "+accuracy",
    "grad_clipping": 1,
    "optimizer": {
      "type": "bert_adam",
      "lr": 0.00005,
      "warmup": 0.1,
      "t_total": 70000,
      "schedule": "warmup_linear"
    }
  }
}
