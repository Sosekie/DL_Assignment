from models.base import BaseModel, BaseImageEncoder, BaseCaptionGenerator


class Model(BaseModel):
    """Base class for all models."""
    def __init__(self, vocabulary):
        super().__init__(vocabulary=vocabulary)


class ImageEncoder(BaseImageEncoder):
    def __init__(self):
        super().__init__()

    def freeze(self):
        """Sets the requires_grad parameter to False for some model parameters."""
        raise NotImplementedError

    def forward(self, image):
        """Forward method.

        :param image: torch.tensor of the shape [batch_size, channels, height, width]

        :return: encoded image (torch.tensor) of the shape [batch_size, *]
        """
        raise NotImplementedError


class CaptionGenerator(BaseCaptionGenerator):
    def __init__(self, vocabulary_size):
        super().__init__()

        self.vocabulary_size = vocabulary_size

    def freeze(self):
        """Sets the requires_grad parameter to False for some model parameters."""
        raise NotImplementedError

    def forward(self, encoded_image, caption_indices, *args):
        """Forward method.

        :param encoded_image: torch.tensor of the shape [batch_size, *] or None
        :param caption_indices: torch.tensor of the shape [batch_size, sequence_length] or None
        :param args: e.g., hidden state

        :return: output dict at least with 'logits' and 'indices' keys,
            where: logits is the torch.tensor of the shape [batch_size, vocabulary_size, sequence_length]
                   indices is the torch.tensor of the shape [batch_size, sequence_length]
        """
        raise NotImplementedError

    def generate_caption_indices(self, encoded_image, sos_token_index, eos_token_index, max_length):
        """Generates caption indices like torch.tensor([1, 23, 5, 8, 2]).

        :param encoded_image: torch.tensor of the shape [1, *]
        :param sos_token_index: index of the "start of sequence" token (int)
        :param eos_token_index: index of the "end of sequence" token (int)
        :param max_length: maximum caption length (int)

        :return: caption indices (list of the length <= max_length)
        """
        raise NotImplementedError
