ELMO=""
DEP_PARSER=""
while getopts ":ed" opt; do
    case $opt in
        e)
            ELMO="-elmo"
            ;;
        d)
            DEP_PARSER="-dp"
            ;;
        \?)
            echo "Invalid option: -$OPTARG" >&2
            ;;
    esac
done

shift $(( OPTIND-1 ))

DATASET="$1"
SERIALIZATION_DIR="./models/se-$DATASET$ELMO$DEP_PARSER"
CONFIG="config/v2/se-$DATASET$ELMO$DEP_PARSER.jsonnet"

#rm -rf "$SERIALIZATION_DIR"
echo "Training $CONFIG. Saving model to $SERIALIZATION_DIR."
allennlp train "$CONFIG" --serialization-dir "$SERIALIZATION_DIR" --include-package syntactic_entailment
