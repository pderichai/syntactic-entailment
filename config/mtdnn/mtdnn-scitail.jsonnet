{
  "dataset_reader": {
    "type": "bert-snli",
    "pretrained_bert_model_file": "pretrained-models/mt-dnn-base",
  },
  "train_data_path": "SciTailV1.1/snli_format/scitail_1.0_train.txt",
  "validation_data_path": "SciTailV1.1/snli_format/scitail_1.0_dev.txt",
  "model": {
    "type": "bert-nli",
    "pretrained_bert_model_file": "pretrained-models/mt-dnn-base",
    "num_labels": 2,
  },
  "iterator": {
    "type": "basic",
    "batch_size": 32
  },
  "trainer": {
    "num_epochs": 5,
    "cuda_device": 0,
    "validation_metric": "+accuracy",
    "grad_clipping": 1,
    "optimizer": {
      "type": "bert_adamax",
      "lr": 0.00005,
      "warmup": 0.1,
      "t_total": 4000,
      "schedule": "warmup_linear"
    }
  }
}
