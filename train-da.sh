SERIALIZATION_DIR="./models/syntactic_entailment"
rm -rf "$SERIALIZATION_DIR"
allennlp train syntactic_entailment.jsonnet --serialization-dir "$SERIALIZATION_DIR" --include-package syntactic_entailment
