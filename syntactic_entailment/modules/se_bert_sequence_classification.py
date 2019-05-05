import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from pytorch_pretrained_bert.modeling import BertPreTrainedModel
from pytorch_pretrained_bert import BertModel


class SyntacticEntailmentBertForSequenceClassification(BertPreTrainedModel):

    def __init__(self, config, num_labels, parser_hidden_size):
        super(SyntacticEntailmentBertForSequenceClassification, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size + 2 * parser_hidden_size, num_labels)
        self.apply(self.init_bert_weights)


    def forward(self, p_encoded_parse, h_encoded_parse, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        _, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        pooled_output = self.dropout(pooled_output)
        pooled_output_with_syntax = torch.cat((pooled_output, p_encoded_parse, h_encoded_parse), dim=-1)
        logits = self.classifier(pooled_output_with_syntax)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        else:
            return logits
