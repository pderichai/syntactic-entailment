python -m allennlp.service.server_simple \
  --archive-path models/se-scitail-demo/model.tar.gz \
  --predictor syntactic-entailment \
  --include-package syntactic_entailment \
  --title Syntactic\ Entailment \
  --field-name premise \
  --field-name hypothesis
