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
TEST_PATH="$2"
SERIALIZATION_DIR="./models/da-$DATASET$ELMO"

echo "Evaluating model at $SERIALIZATION_DIR with $TEST_PATH."
allennlp evaluate "$SERIALIZATION_DIR"/model.tar.gz $TEST_PATH
