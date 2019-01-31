import torch
from overrides import overrides

from allennlp.modules.similarity_functions.dot_product import DotProductSimilarity
from allennlp.modules.similarity_functions.similarity_function import SimilarityFunction
from allennlp.modules.matrix_attention.matrix_attention import MatrixAttention


@MatrixAttention.register("syntactic")
class SyntacticMatrixAttention(MatrixAttention):
    """
    Overriding shit.

    Parameters
    ----------
    similarity_function: ``SimilarityFunction``, optional (default=``DotProductSimilarity``)
        The similarity function to use when computing the attention.
    """
    def __init__(self, similarity_function: SimilarityFunction = None) -> None:
        super().__init__()
        self._similarity_function = similarity_function or DotProductSimilarity()

    @overrides
    def forward(self, matrix_1: torch.Tensor, matrix_2: torch.Tensor,
                syntax_1: torch.Tensor, syntax_2: torch.Tensor) -> torch.Tensor:
        #print('syntax_1', syntax_1.is_cuda, syntax_1.shape)
        #print('matrix_1', matrix_1.is_cuda, matrix_1.shape)
        matrix_1 = torch.cat((matrix_1, syntax_1), 2)
        matrix_2 = torch.cat((matrix_2, syntax_2), 2)
        #print('catted matrix_1', matrix_1.shape)
        tiled_matrix_1 = matrix_1.unsqueeze(2).expand(matrix_1.size()[0],
                                                      matrix_1.size()[1],
                                                      matrix_2.size()[1],
                                                      matrix_1.size()[2])
        #print('tiled_matrix_1', tiled_matrix_1.shape)
        #print('syntax_2', syntax_2.shape)
        #print('matrix_2', matrix_2.shape)
        tiled_matrix_2 = matrix_2.unsqueeze(1).expand(matrix_2.size()[0],
                                                      matrix_1.size()[1],
                                                      matrix_2.size()[1],
                                                      matrix_2.size()[2])
        #print('tiled_matrix_2', tiled_matrix_2.shape)
        ret = self._similarity_function(tiled_matrix_1, tiled_matrix_2)
        #print('ret', ret.shape)
        return ret
