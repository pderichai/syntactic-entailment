from torch.nn import CrossEntropyLoss
from pytorch_pretrained_bert import BertForSequenceClassification


class SyntacticEntailmentBertForSequenceClassification(BertForSequenceClassification):

    def __init__(self, config, num_labels):
        super(SyntacticEntailmentBertForSequenceClassification, self).__init__(config, num_labels)


    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        _, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        else:
            return logits
