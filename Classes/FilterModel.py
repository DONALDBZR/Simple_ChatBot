from Classes.ResponseFilterNeuralNetwork import Response_Filter_Neural_Network
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from typing import List
from torch import tensor, Tensor, float32, long, no_grad
from numpy import float64, float32 as np_float_32
from numpy.typing import NDArray


class Filter_Model:
    """
    It uses a neural network to filter input data based on their embeddings.

    Attributes:
        __model (Response_Filter_Neural_Network): The neural network used for filtering.
        __optimizer (Adam): The optimizer used for training the model.
        __loss_function (CrossEntropyLoss): The loss function used for training the model.

    Methods:
        trainFilter(embeddings: NDArray[float64], labels: List[int], epochs: int) -> None: Training the filter model for a given number of epochs.
        score(self, embeddings: NDArray[float64]) -> NDArray[np_float_32]: Scoring the given embeddings with the filter model.
    """
    __model: Response_Filter_Neural_Network
    __optimizer: Adam
    __loss_function: CrossEntropyLoss

    def __init__(
        self,
        input_size: int = 100,
        learning_rate: float = 1e-3
    ):
        """
        Initializing the `FilterModel` with a given input size and learning rate.

        Args:
            input_size (int, optional): The size of the input. Defaults to 100.
            learning_rate (float, optional): The learning rate for the optimizer. Defaults to 1e-3.
        """
        self.model = Response_Filter_Neural_Network(input_size)
        self.optimizer = Adam(
            params=self.model.parameters(),
            lr=learning_rate
        )
        self.loss_function = CrossEntropyLoss()

    @property
    def model(self) -> Response_Filter_Neural_Network:
        return self.__model

    @model.setter
    def model(self, model: Response_Filter_Neural_Network) -> None:
        self.__model = model

    @property
    def optimizer(self) -> Adam:
        return self.__optimizer

    @optimizer.setter
    def optimizer(self, optimizer: Adam) -> None:
        self.__optimizer = optimizer

    @property
    def loss_function(self) -> CrossEntropyLoss:
        return self.__loss_function

    @loss_function.setter
    def loss_function(self, loss_function: CrossEntropyLoss) -> None:
        self.__loss_function = loss_function

    def trainFilter(
        self,
        embeddings: NDArray[float64],
        labels: List[int],
        epochs: int = 10
    ) -> None:
        """
        Training the filter model with the given embeddings and labels.

        Args:
            embeddings (NDArray[float64]): The embeddings to train the filter model with.
            labels (List[int]): The labels of the embeddings.
            epochs (int, optional): The number of epochs to train the filter model. Defaults to 10.
        """
        horizontal_tensor: Tensor = tensor(
            data=embeddings,
            dtype=float32
        )
        vertical_tensor: Tensor = tensor(
            data=labels,
            dtype=long
        )
        self.model.train()
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            output: Tensor = self.model(horizontal_tensor)
            loss_data: Tensor = self.loss_function(output, vertical_tensor)
            loss_data.backward()
            self.optimizer.step()

    def score(self, embeddings: NDArray[float64]) -> NDArray[np_float_32]:
        """
        Scoring the given embeddings with the filter model.

        Args:
            embeddings (NDArray[float64]): The embeddings to score.

        Returns:
            NDArray[np_float_32]: The scores of the given embeddings.
        """
        horizontal_tensor: Tensor = tensor(
            data=embeddings,
            dtype=float32
        )
        self.model.eval()
        with no_grad():
            scores: NDArray[np_float_32] = self.model(horizontal_tensor)[:,1].numpy()
        return scores
