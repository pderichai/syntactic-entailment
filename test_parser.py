from allennlp.predictors.predictor import Predictor
predictor = Predictor.from_path("models/elmo-constituency-parser-2018.03.14.tar.gz")
p = predictor.predict(sentence="If I bring 10 dollars tomorrow, can you buy me lunch?")
print(p.keys())
print(p['hierplane_tree'])
