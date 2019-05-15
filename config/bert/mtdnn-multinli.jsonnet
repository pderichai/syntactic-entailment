{
  "dataset_reader": {
    "type": "bert-snli",
    "pretrained_bert_model_file": "pretrained-models/mt-dnn-base",
  },
  "train_data_path": "multinli_1.0/multinli_1.0_train.jsonl",
  "validation_data_path": "multinli_1.0/multinli_1.0_dev_matched.jsonl",
  "model": {
    "type": "bert-nli",
    "pretrained_bert_model_file": "pretrained-models/mt-dnn-base",
    "num_labels": 3,
  },
  "iterator": {
    "type": "basic",
    "batch_size": 32
  },
  "trainer": {
    "num_epochs": 5,
    "cuda_device": 0,
    "validation_metric": "+accuracy",
    "optimizer": {
      "type": "bert_adamax",
      "lr": 0.00005,
      "warmup": 0.1,
      "t_total": 10000,
      "schedule": "warmup_linear"
    }
  }
}
