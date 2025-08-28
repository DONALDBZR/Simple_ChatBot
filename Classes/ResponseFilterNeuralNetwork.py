from torch.nn import Module, Linear
from torch import Tensor, relu


class Response_Filter_Neural_Network(Module):
    """
    It is a neural network for filtering the responses.

    Attributes:
        fully_connected_layer_1 (Linear): The first fully connected layer.
        fully_connected_layer_2 (Linear): The second fully connected layer.
        fully_connected_layer_3 (Linear): The third fully connected layer.

    Methods:
        forward(self, input_tensor: Tensor) -> Tensor: Forwarding the input tensor propagation through the neural network.
    """
    __fully_connected_layer_1: Linear
    __fully_connected_layer_2: Linear
    __fully_connected_layer_3: Linear

    def __init__(
        self,
        input_size: int = 100
    ):
        """
        Initializing the `Response_Filter_Neural_Network` with a given input size.

        Args:
            input_size (int, optional): The size of the input. Defaults to 100.
        """
        super().__init__()
        self.fully_connected_layer_1 = Linear(input_size, 128)
        self.fully_connected_layer_2 = Linear(128, 64)
        self.fully_connected_layer_3 = Linear(64, 2)

    @property
    def fully_connected_layer_1(self) -> Linear:
        return self.__fully_connected_layer_1

    @fully_connected_layer_1.setter
    def fully_connected_layer_1(self, value: Linear) -> None:
        self.__fully_connected_layer_1 = value

    @property
    def fully_connected_layer_2(self) -> Linear:
        return self.__fully_connected_layer_2

    @fully_connected_layer_2.setter
    def fully_connected_layer_2(self, value: Linear) -> None:
        self.__fully_connected_layer_2 = value

    @property
    def fully_connected_layer_3(self) -> Linear:
        return self.__fully_connected_layer_3

    @fully_connected_layer_3.setter
    def fully_connected_layer_3(self, value: Linear) -> None:
        self.__fully_connected_layer_3 = value

    def forward(self, input_tensor: Tensor) -> Tensor:
        """
        Forwarding the input tensor propagation through the neural network.

        Args:
            input_tensor (Tensor): The tensor to be propagated through the network.

        Returns:
            Tensor: The output of the network.
        """
        return self.fully_connected_layer_3(
            relu(
                self.fully_connected_layer_2(
                    relu(
                        self.fully_connected_layer_1(
                            input_tensor
                        )
                    )
                )
            )
        )
