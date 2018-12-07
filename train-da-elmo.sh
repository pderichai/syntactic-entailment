SERIALIZATION_DIR="./models/syntactic-entailment-elmo"
rm -rf "$SERIALIZATION_DIR"
allennlp train syntactic_entailment_elmo.jsonnet --serialization-dir "$SERIALIZATION_DIR" --include-package syntactic_entailment
