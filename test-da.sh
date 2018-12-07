SERIALIZATION_DIR="./models/syntactic-entailment"
allennlp evaluate "$SERIALIZATION_DIR"/model.tar.gz SciTailV1.1/snli_format/scitail_1.0_test.txt --include-package syntactic_entailment
