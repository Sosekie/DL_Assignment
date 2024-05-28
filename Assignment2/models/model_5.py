import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models
from typing import Tuple

from einops import rearrange, pack

from models.base import BaseModel, BaseImageEncoder, BaseCaptionGenerator

if_print = False

class Model(BaseModel):
    def __init__(self, vocabulary, embedding_dim, num_layers):
        super().__init__(vocabulary=vocabulary)

        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.backbone = DINOv2Backbone()

        self.image_encoder = ImageEncoder(backbone=self.backbone, embedding_dim=self.embedding_dim)
        self.caption_generator = CaptionGenerator(vocabulary_size=len(self.vocabulary),
                                                  embedding_dim=self.embedding_dim,
                                                  hidden_dim=self.embedding_dim,
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
        self.relu = torch.nn.ReLU()

    # model.freeze() is called in train.py
    def freeze(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def forward(self, image):
        """
        :param x: [b c h w]
        """

        feats = self.backbone(image)
        out = self.out(feats[0])

        # You can choose using relu or not, here I did not use it        
        return out


class CaptionGenerator(BaseCaptionGenerator):
    def __init__(self, vocabulary_size, embedding_dim, hidden_dim, num_layers):
        super().__init__(vocabulary_size=vocabulary_size)

        self.embedding_dim = embedding_dim

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.embedding = torch.nn.Sequential(torch.nn.Embedding(num_embeddings=self.vocabulary_size,
                                                                embedding_dim=self.embedding_dim),
                                             torch.nn.Dropout(0.5))
        
        self.transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=self.embedding_dim,
                                                                    nhead=8,
                                                                    dim_feedforward=hidden_dim,
                                                                    dropout=0.1,
                                                                    activation='relu')
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_encoder_layer, num_layers=self.num_layers)
        
        self.to_logits = torch.nn.Linear(in_features=self.embedding_dim, out_features=self.vocabulary_size)

    def freeze(self):
        pass

    def _get_embeddings(self, encoded_image=None, caption_indices=None):
        if caption_indices is None:
            embeddings = rearrange(encoded_image, 'batch embedding_dim -> batch 1 embedding_dim')
        else:
            embeddings = self.embedding(caption_indices)
            if if_print:
                print('embeddings before size: ', embeddings.size())
            if encoded_image is not None:
                embeddings, _ = pack([encoded_image, embeddings], 'batch * embedding_dim')

        return embeddings
    
    def generate_square_subsequent_mask(self, size, device):
        mask = torch.triu(torch.full((size, size), float('-inf'), device=device), 1)
        return mask

    def forward(self, encoded_image, caption_indices, hidden_state=None):

        if caption_indices is not None and if_print:
            print('caption_indices size: ', caption_indices.size())

        if encoded_image is not None and caption_indices is not None:
            caption_indices = caption_indices[:, 1:]  # the encoded image will be used instead of the <SOS> token
        
        if caption_indices is not None and if_print:
            print('caption_indices size: ', caption_indices.size())

        embeddings = self._get_embeddings(encoded_image=encoded_image, caption_indices=caption_indices)

        if if_print:
            print('embeddings size: ', embeddings.size())

        # expects input of shape (sequence_length, batch_size, embedding_dim)
        embeddings = rearrange(embeddings, 'batch seq_len embedding_dim -> seq_len batch embedding_dim')

        # generate mask, which is the most important thing
        src_mask = self.generate_square_subsequent_mask(embeddings.size(0), embeddings.device)

        output = self.transformer_encoder(embeddings, mask = src_mask)
        
        # convert back to shape (batch_size, sequence_length, embedding_dim)
        output = rearrange(output, 'seq_len batch embedding_dim -> batch seq_len embedding_dim')

        logits = self.to_logits(output)

        if if_print:
            print('logits size: ', logits.size())

        logits = rearrange(logits, 'batch sequence_length vocabulary_size -> batch vocabulary_size sequence_length')

        if if_print:
            print('logits size: ', logits.size())

        return {'logits': logits, 'indices': logits.argmax(dim=-2), 'hidden_state': hidden_state}

    def generate_caption_indices(self, encoded_image, sos_token_index, eos_token_index, max_length):
        caption_indices = [sos_token_index]
        caption_tensor = torch.tensor(caption_indices, device=encoded_image.device).unsqueeze(0)

        for _ in range(max_length):
            output = self.forward(encoded_image, caption_tensor)

            predicted_index = output['indices'][:, -1]

            caption_indices.append(predicted_index.item())

            if predicted_index.item() == eos_token_index:
                break

            caption_tensor = torch.tensor(caption_indices, device=encoded_image.device).unsqueeze(0)

        return caption_indices
