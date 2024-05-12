import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models
from typing import Tuple

from einops import rearrange, pack, repeat

from models.base import BaseModel, BaseImageEncoder, BaseCaptionGenerator


class Model(BaseModel):
    def __init__(self, vocabulary, embedding_dim, num_layers):
        super().__init__(vocabulary=vocabulary)

        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.backbone = DINOv2Backbone()

        self.image_encoder = ImageEncoder(backbone=self.backbone, embedding_dim=self.embedding_dim)
        self.caption_generator = CaptionGenerator(vocabulary_size=len(self.vocabulary),
                                                  embedding_dim=self.embedding_dim,
                                                  num_layers=self.num_layers)


class DINOv2Backbone(nn.Module):
    def __init__(self):
        super(DINOv2Backbone, self).__init__()
        
        self.dino = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        self.embed_dim = self.dino.embed_dim
    
    def forward(self, x: torch.Tensor, scale: int = 1) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :param x: [b c h w]
        """

        x = F.interpolate(x, size=(scale * 224, scale * 224), mode="bilinear", align_corners=False)
        out = self.dino.get_intermediate_layers(x, n=1, reshape=True, return_class_token=True)[0]

        return out[1], out[0]
    

class ImageEncoder(nn.Module):
    def __init__(self, backbone: nn.Module, embedding_dim: int):
        super().__init__()

        self.backbone = backbone
        self.out = nn.Sequential(
            nn.LayerNorm(self.backbone.embed_dim),
            nn.Linear(self.backbone.embed_dim, embedding_dim)
        )

    def freeze(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def forward(self, image):
        feats = self.backbone(image)
        out = self.out(feats[0]) 
        return out


class CaptionGenerator(BaseCaptionGenerator):
    def __init__(self, vocabulary_size, embedding_dim, num_layers):
        super().__init__(vocabulary_size=vocabulary_size)
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers

        self.embedding = nn.Embedding(num_embeddings=self.vocabulary_size, embedding_dim=self.embedding_dim)
        self.positional_encoding = nn.Parameter(torch.zeros(1, 200, self.embedding_dim))  # Assuming max sequence length is 200
        self.transformer = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=self.embedding_dim, nhead=8), num_layers=self.num_layers)
        self.to_logits = nn.Linear(in_features=self.embedding_dim, out_features=self.vocabulary_size)

    def freeze(self):
        pass

    def _get_embeddings(self, encoded_image=None, caption_indices=None):
        if caption_indices is None:
            embeddings = rearrange(encoded_image, 'batch embedding_dim -> batch 1 embedding_dim')
        else:
            embeddings = self.embedding(caption_indices)
            if encoded_image is not None:
                embeddings, _ = pack([encoded_image, embeddings], 'batch * embedding_dim')

        return embeddings

    def forward(self, encoded_image, caption_indices):
        if encoded_image is not None and caption_indices is not None:
            caption_indices = caption_indices[:, 1:]  # the encoded image will be used instead of the <SOS> token

        embeddings = self._get_embeddings(encoded_image=encoded_image, caption_indices=caption_indices)

        transformer_output = self.transformer(embeddings)
        logits = self.to_logits(transformer_output)
        logits = rearrange(logits, 'batch sequence_length vocabulary_size -> batch vocabulary_size sequence_length')

        return {'logits': logits, 'indices': logits.argmax(dim=-2)}
    
    def generate_caption_indices(self, encoded_image, sos_token_index, eos_token_index, max_length):
        caption_indices = []

        output = self.forward(encoded_image, caption_indices=None)
        for _ in range(max_length):
            predicted_index = output['indices']

            caption_indices.append(predicted_index.item())
            if predicted_index.item() == eos_token_index:
                break

            output = self.forward(encoded_image=None,
                                  caption_indices=predicted_index)

        return caption_indices
