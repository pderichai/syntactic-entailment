{
  "dataset_reader": {
    "type": "bert-snli",
    "pretrained_bert_model_file": "bert-base-uncased",
  },
  "train_data_path": "SciTailV1.1/snli_format/scitail_1.0_train.txt",
  "validation_data_path": "SciTailV1.1/snli_format/scitail_1.0_dev.txt",
  "model": {
    "type": "bert-nli",
    "pretrained_bert_model_file": "bert-base-uncased",
    "num_labels": 2,
  },
  "iterator": {
    "type": "basic",
    "batch_size": 40
  },
  "trainer": {
    "num_epochs": 3,
    "cuda_device": 0,
    "validation_metric": "+accuracy",
    "optimizer": {
      "type": "adam",
      "lr": 0.00005
    }
  }
}
