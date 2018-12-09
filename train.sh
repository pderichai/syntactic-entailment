DATASET="$1"
ELMO=""
while getopts ":e" opt; do
    case $opt in
        e)
            ELMO="-elmo"
            ;;
        \?)
            echo "Invalid option: -$OPTARG" >&2
            ;;
    esac
done

SERIALIZATION_DIR="./models/se-$DATASET$ELMO"
CONFIG="se-$DATASET$ELMO.jsonnet"
echo "Training $CONFIG. Saving model to $SERIALIZATION_DIR."

rm -rf "$SERIALIZATION_DIR"
allennlp train "$CONFIG" --serialization-dir "$SERIALIZATION_DIR" --include-package syntactic_entailment
