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

shift $(( OPTIND-1 ))

DATASET="$1"
SERIALIZATION_DIR="./models/se-$DATASET$ELMO"
CONFIG="config/se-$DATASET$ELMO.jsonnet"

rm -rf "$SERIALIZATION_DIR"
echo "Training $CONFIG. Saving model to $SERIALIZATION_DIR."
allennlp train "$CONFIG" --serialization-dir "$SERIALIZATION_DIR" --include-package syntactic_entailment
