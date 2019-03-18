ELMO=""
DEP_PARSER=""
CUDA_DEVICE=""
while getopts ":edc" opt; do
    case $opt in
        e)
            ELMO="-elmo"
            ;;
        d)
            DEP_PARSER="-dependency-parser"
            ;;
        c)
            CUDA_DEVICE="--cuda-device 0"
            ;;
        \?)
            echo "Invalid option: -$OPTARG" >&2
            ;;
    esac
done

shift $(( OPTIND-1 ))

DATASET="$1"
TEST_PATH="$2"
SERIALIZATION_DIR="./models/se-$DATASET$ELMO$DEP_PARSER"

echo "Evaluating model at $SERIALIZATION_DIR with $TEST_PATH."
allennlp evaluate "$SERIALIZATION_DIR"/model.tar.gz $TEST_PATH --include-package syntactic_entailment $CUDA_DEVICE
