from nltk.corpus import wordnet
from pydictionary import Dictionary
from typing import List, Optional, Dict
from nltk.corpus.reader.wordnet import Synset


class Dictionary_Service:
    """
    It is the service that will act as dictionary.

    Attributes:
        dictionary: (Dictionary): The dictionary that will be used to find the definition of the given word.

    Methods:
        defineWord(word: str) -> str: Retrieving the definition of the given word.
    """
    __dictionary: Dictionary

    def __init__(self):
        pass

    @property
    def dictionary(self) -> Dictionary:
        return self.__dictionary

    @dictionary.setter
    def dictionary(self, dictionary: Dictionary) -> None:
        self.__dictionary = dictionary

    def defineWord(self, word: str) -> str:
        """
        Retrieving the definition of the given word.

        Args:
            word (str): The word for which the definition will be retrieved.

        Returns:
            str: The definition of the given word, or "No definition found." if no definition was found.
        """
        self.dictionary = Dictionary(word)
        definitions: Optional[Dict[str, List[str]]] = self.dictionary.meanings() # type: ignore
        if definitions:
            return "; ".join(f"{key}: {', '.join(value)}" for key, value in definitions.items())
        synsets: List[Synset] = wordnet.synsets(word) # type: ignore
        if synsets:
            return "; ".join(set(synset.definition() for synset in synsets)) # type: ignore
        return "No definition found."
