from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from numpy import ndarray, zeros, hstack, float64
from typing import Union, Tuple, Any, List
from scipy.sparse import spmatrix
from numpy.typing import NDArray


class Text_Embedder:
    """
    It embeds text data unto numerical vectors by using a Term Frequency-Inverse Document Frequency and Singular Value Decomposition.

    Attributes:
        vectorizer (TfidfVectorizer): The term frequency-inverse document frequency vectorizer to be used for embedding.

    Methods:
        embed(texts: ndarray, dimensionality: int = 100) -> ndarray: Embedding the given texts using the vectorizer and `TruncatedSVD` to the given dimensionality.
    """
    __vectorizer: TfidfVectorizer

    def __init__(self):
        """
        Initializing the vectorizer to a default TfidfVectorizer object.
        """
        self.vectorizer = TfidfVectorizer()

    @property
    def vectorizer(self) -> TfidfVectorizer:
        return self.__vectorizer

    @vectorizer.setter
    def vectorizer(self, value: TfidfVectorizer) -> None:
        self.__vectorizer = value

    def embed(
        self,
        texts: List[str],
        dimensionality: int = 100
    ) -> NDArray[float64]:
        """
        Embedding the given texts using the vectorizer and `TruncatedSVD` to the given dimensionality.

        Args:
            texts (List[str]): The texts to be embedded.
            dimensionality (int, optional): The dimensionality to be used for the embedding. Defaults to 100.

        Returns:
            NDArray[float64]: The embedded vectors.
        """
        vectors: spmatrix = self.vectorizer.fit_transform(texts)
        features: int = vectors.shape[1]
        components: int = min(dimensionality, features)
        singular_value_decomposition: TruncatedSVD = TruncatedSVD(
            n_components=components
        )
        reduced: NDArray[float64] = singular_value_decomposition.fit_transform(vectors)
        if components >= dimensionality:
            return reduced
        padding: NDArray[float64] = zeros(
            shape=(reduced.shape[0], dimensionality - components),
            dtype=float64
        )
        reduced = hstack((reduced, padding))
        return reduced
