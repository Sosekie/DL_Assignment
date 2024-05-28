import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, pack
from models.base import BaseModel, BaseImageEncoder, BaseCaptionGenerator

if_print = False
position_encoding = False
attention_map_visualize = True
using_cls = False

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
    
    # overwrite base.py's generate_image_caption_tokens to generate attention map
    if attention_map_visualize:
        def generate_image_caption_tokens(self, image, max_length=50):
            self.image_encoder.eval()
            self.caption_generator.eval()

            eos_token_index = self.vocabulary.to_index(self.vocabulary.eos_token)
            sos_token_index = self.vocabulary.to_index(self.vocabulary.sos_token)
            with torch.no_grad():
                encoded_image = self.image_encoder.forward(image=image)
                caption_indices, attention_maps = self.caption_generator.generate_caption_indices(encoded_image=encoded_image,
                                                                                                sos_token_index=sos_token_index,
                                                                                                eos_token_index=eos_token_index,
                                                                                                max_length=max_length)

            caption_tokens = self.vocabulary.to_tokens(indices=caption_indices, remove_special_tokens=True)

            print('New function!')

            return caption_tokens, attention_maps


class DINOv2Backbone(nn.Module):
    def __init__(self):
        super(DINOv2Backbone, self).__init__()
        
        self.dino = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        self.embed_dim = self.dino.embed_dim
    
    def forward(self, x: torch.Tensor, scale: int = 1) -> torch.Tensor:
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

    def freeze(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def forward(self, image):
        feats = self.backbone(image)
        out_cls = self.out(feats[0])

        # feats[1]: [batch, embedding_dim, height, width] -> [batch, height * width, embedding_dim]
        batch_size, embed_dim, height, width = feats[1].shape
        out_feats = feats[1].permute(0, 2, 3, 1).reshape(batch_size, height * width, embed_dim)
        out_feats = self.out(out_feats)

        return [out_feats, out_cls]


class CaptionGenerator(BaseCaptionGenerator):
    def __init__(self, vocabulary_size, embedding_dim, hidden_dim, num_layers):
        super().__init__(vocabulary_size=vocabulary_size)

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.embedding = torch.nn.Sequential(torch.nn.Embedding(num_embeddings=self.vocabulary_size,
                                                                embedding_dim=self.embedding_dim),
                                             torch.nn.Dropout(0.5))
        
        if position_encoding:
            self.positional_encoding_image = nn.Parameter(torch.zeros(1, 256, embedding_dim))
            self.positional_encoding_caption = nn.Parameter(torch.zeros(1, 100, embedding_dim))
        
        self.transformer_decoder_layer = nn.TransformerDecoderLayer(d_model=self.embedding_dim,
                                                                    nhead=8,
                                                                    dim_feedforward=hidden_dim,
                                                                    dropout=0.1,
                                                                    activation='relu')
        self.transformer_decoder = nn.TransformerDecoder(self.transformer_decoder_layer, num_layers=self.num_layers)
        
        self.to_logits = torch.nn.Linear(in_features=self.embedding_dim, out_features=self.vocabulary_size)

    def freeze(self):
        pass

    def _get_embeddings(self, cls_token=None, caption_indices=None):
        if caption_indices is None:
            embeddings = rearrange(cls_token, 'batch embedding_dim -> batch 1 embedding_dim')
        else:
            embeddings = self.embedding(caption_indices)
            if if_print:
                print('embeddings before size: ', embeddings.size())
            if cls_token is not None:
                embeddings, _ = pack([cls_token, embeddings], 'batch * embedding_dim')

        return embeddings
    
    def generate_square_subsequent_mask(self, size, device):
        mask = torch.triu(torch.full((size, size), float('-inf'), device=device), 1)
        return mask

    def forward(self, encoded_image, caption_indices, hidden_state=None):
        patch_token, cls_token = encoded_image

        if not using_cls:
            cls_token = None

        if caption_indices is not None and if_print:
            print('caption_indices size: ', caption_indices.size())

        if cls_token is not None and caption_indices is not None:
            caption_indices = caption_indices[:, 1:]  # the encoded image will be used instead of the <SOS> token
        
        if caption_indices is not None and if_print:
            print('caption_indices size: ', caption_indices.size())

        embeddings = self._get_embeddings(cls_token=cls_token, caption_indices=caption_indices)

        if if_print:
            print('embeddings size: ', embeddings.size())

        # expects input of shape (sequence_length, batch_size, embedding_dim)
        embeddings = rearrange(embeddings, 'batch seq_len embedding_dim -> seq_len batch embedding_dim')

        seq_len, batch_size, _ = embeddings.size()
        if position_encoding:
            # add positional encoding to embeddings
            pos_encoding = self.positional_encoding_caption[:, :seq_len, :].expand(batch_size, -1, -1)
            pos_encoding = rearrange(pos_encoding, 'batch seq_len embedding_dim -> seq_len batch embedding_dim')
            embeddings += pos_encoding

        # generate mask
        tgt_mask = self.generate_square_subsequent_mask(seq_len, embeddings.device)

        batch_size, num_patches, embed_dim = patch_token.shape
        memory = rearrange(patch_token, 'batch seq_len embedding_dim -> seq_len batch embedding_dim')

        if if_print:
            print('patch_token size: ', patch_token.size())
            print('memory size: ', memory.size())

        if position_encoding:
            # Add positional encoding to memory
            pos_enc_image = self.positional_encoding_image[:, :num_patches, :].expand(batch_size, -1, -1)
            pos_enc_image = rearrange(pos_enc_image, 'batch seq_len embedding_dim -> seq_len batch embedding_dim')
            memory += pos_enc_image

        output = self.transformer_decoder(embeddings, memory, tgt_mask=tgt_mask)

        if attention_map_visualize:
            attention_weights = self.transformer_decoder_layer.self_attn(embeddings, embeddings, embeddings)[1]
        
        # convert back to shape (batch_size, sequence_length, embedding_dim)
        output = rearrange(output, 'seq_len batch embedding_dim -> batch seq_len embedding_dim')

        logits = self.to_logits(output)

        if if_print:
            print('logits size: ', logits.size())

        logits = rearrange(logits, 'batch sequence_length vocabulary_size -> batch vocabulary_size sequence_length')

        if if_print:
            print('logits size: ', logits.size())

        if attention_map_visualize:
            return {'logits': logits, 'indices': logits.argmax(dim=-2), 'attention_weights': attention_weights}
        else:
            return {'logits': logits, 'indices': logits.argmax(dim=-2), 'hidden_state': hidden_state}

    def generate_caption_indices(self, encoded_image, sos_token_index, eos_token_index, max_length):
        caption_indices = [sos_token_index]
        caption_tensor = torch.tensor(caption_indices, device=encoded_image[0].device).unsqueeze(0)

        if attention_map_visualize:
            attention_maps = []

        for _ in range(max_length):
            tgt_mask = self.generate_square_subsequent_mask(caption_tensor.size(1), caption_tensor.device)
            output = self.forward(encoded_image, caption_tensor, hidden_state=None)

            predicted_index = output['indices'][:, -1]

            if attention_map_visualize:
                attention_map = output['attention_weights'][:, -1, :].squeeze()

            caption_indices.append(predicted_index.item())

            if attention_map_visualize:
                attention_maps.append(attention_map)

            if predicted_index.item() == eos_token_index:
                break

            caption_tensor = torch.tensor(caption_indices, device=encoded_image[0].device).unsqueeze(0)

        if attention_map_visualize:
            return caption_indices, attention_maps
        else:
            return caption_indices