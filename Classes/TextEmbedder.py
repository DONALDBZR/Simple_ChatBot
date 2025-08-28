import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from numpy import ndarray, zeros, hstack
from typing import Union, Tuple, Any


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
        texts: ndarray[Any, Any],
        dimensionality: int = 100
    ) -> ndarray:
        """
        Embedding the given texts using the vectorizer and `TruncatedSVD` to the given dimensionality.

        Args:
            texts (ndarray): The texts to be embedded.
            dimensionality (int): The number of dimensions to reduce to. Defaults to 100.

        Returns:
            ndarray: The embedded vectors.
        """
        vectors: Union[Tuple[Any], Any] = self.vectorizer.fit_transform(texts)
        features = vectors.shape[1]
        components = min(dimensionality, features)
        singular_value_decomposition: TruncatedSVD = TruncatedSVD(
            n_components=components
        )
        reduced: ndarray = singular_value_decomposition.fit_transform(vectors)
        if components >= dimensionality:
            return reduced
        padding: ndarray = zeros((reduced.shape[0], dimensionality - components))
        reduced = hstack((reduced, padding))
        return reduced
