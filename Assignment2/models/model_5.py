import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models
from typing import Tuple
import math

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
                                                  num_heads=8,
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


class CaptionGenerator(nn.Module):
    def __init__(self, vocabulary_size, embedding_dim, num_heads, num_layers):
        super().__init__()

        self.vocabulary_size = vocabulary_size
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.num_layers = num_layers

        self.embedding = nn.Embedding(num_embeddings=vocabulary_size, embedding_dim=self.embedding_dim)
        self.positional_encoding = self._generate_positional_encoding(self.embedding_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embedding_dim,
            nhead=self.num_heads,
            dropout=0.5,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)

        self.to_logits = torch.nn.Linear(in_features=self.embedding_dim, out_features=self.vocabulary_size)

    def freeze(self):
        pass

    def _generate_positional_encoding(self, dim, max_len=200):
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    def _get_embeddings(self, encoded_image=None, caption_indices=None):
        embeddings = self.embedding(caption_indices)
        embeddings = embeddings + self.positional_encoding[:embeddings.size(1)].to(embeddings.device)
        encoded_image = rearrange(encoded_image, 'batch embedding_dim -> batch 1 embedding_dim')
        embeddings = torch.cat([encoded_image, embeddings], dim=1)

        return embeddings

    def forward(self, encoded_image, caption_indices):
        if encoded_image is not None and caption_indices is not None:
            caption_indices = caption_indices[:, 1:]  # the encoded image will be used instead of the <SOS> token

        embeddings = self._get_embeddings(encoded_image=encoded_image, caption_indices=caption_indices)
        
        output = self.transformer_encoder(embeddings)
        logits = self.to_logits(output)
        logits = rearrange(logits, 'batch sequence_length vocabulary_size -> batch vocabulary_size sequence_length')

        return {'logits': logits, 'indices': logits.argmax(dim=-2)}
    
    def generate_caption_indices(self, encoded_image, sos_token_index, eos_token_index, max_length):
        caption_indices = [sos_token_index]

        for _ in range(max_length):
            current_indices = torch.tensor([caption_indices], dtype=torch.long, device=encoded_image.device)
            output = self.forward(encoded_image, current_indices)
            predicted_index = output['indices'][0, -1].item()  # get the last predicted index

            caption_indices.append(predicted_index)
            if predicted_index == eos_token_index:
                break

        return caption_indices
