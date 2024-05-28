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
        
        # double the input size, since each step we consider encoded_image 
        # mention that now we can use <SOS> token as start, so the input size is still embedding_dim*2
        self.lstm = torch.nn.LSTM(input_size=self.embedding_dim + self.embedding_dim, 
                                  hidden_size=self.hidden_dim,
                                  num_layers=self.num_layers, bias=True, batch_first=True)
        
        self.to_logits = torch.nn.Linear(in_features=self.hidden_dim, out_features=self.vocabulary_size)

    def freeze(self):
        pass

    def _get_embeddings(self, encoded_image=None, caption_indices=None):
        # now we do not need to justify whether the first token is from image or text
        # Since it is a combination of both image and text(maybe sos_token)
        text_embeddings  = self.embedding(caption_indices)
        image_embeddings = repeat(encoded_image, 'batch embedding_dim -> batch seq_len embedding_dim', seq_len=text_embeddings.shape[1])
        embeddings = torch.cat([image_embeddings, text_embeddings], dim=-1)

        return embeddings

    def forward(self, encoded_image, caption_indices, hidden_state=None):
        embeddings = self._get_embeddings(encoded_image=encoded_image, caption_indices=caption_indices)
        output, hidden_state = self.lstm(input=embeddings, hx=hidden_state)
        logits = self.to_logits(output)
        logits = rearrange(logits, 'batch sequence_length vocabulary_size -> batch vocabulary_size sequence_length')

        return {'logits': logits, 'indices': logits.argmax(dim=-2), 'hidden_state': hidden_state}

    def generate_caption_indices(self, encoded_image, sos_token_index, eos_token_index, max_length):
        device = encoded_image.device
        caption_indices = [sos_token_index]
        caption_indices_tensor = torch.tensor(caption_indices, dtype=torch.long, device=device).unsqueeze(0)

        output = self.forward(encoded_image, caption_indices=caption_indices_tensor, hidden_state=None)
        for _ in range(max_length):
            predicted_index = output['indices']

            caption_indices.append(predicted_index.item())
            if predicted_index.item() == eos_token_index:
                break
            
            # Now each step we should using encoded_image for conditioning
            output = self.forward(encoded_image=encoded_image,
                                  caption_indices=predicted_index,
                                  hidden_state=output['hidden_state'])

        return caption_indices
