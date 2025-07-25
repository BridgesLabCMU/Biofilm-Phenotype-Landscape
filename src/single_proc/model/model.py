from .utils import *
from torch import nn


def PositionalEncoding(embeddings, max_len=128):
    model_dims = embeddings.shape[1]


    two_i = torch.arange(0, model_dims, 2, dtype=torch.float32, device=device)

    div_term = 1 / (10000 ** (two_i / max_len))
    position = torch.arange(0,max_len,dtype=torch.float, device=device).unsqueeze(1)

    encoding = torch.zeros(max_len, model_dims, device=device)
    encoding[:,0::2] = torch.sin(position * div_term)
    encoding[:,1::2] = torch.cos(position * div_term)
    image_embeddings = torch.add(embeddings, encoding[:embeddings.shape[0], :embeddings.shape[1]])
    
    return torch.mean(image_embeddings, axis=0)
        
class ViT(nn.Module):
    def __init__(self, model, num_frames=25, embedding_dims=128):
        """
        Loads Vision Transformer from An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale.
        
        Arguments:
        num_frames - number of frames in input video
        embedding_dims - output embedding dimension 
        """
        super(ViT, self).__init__()

        self.model = model
        
        self.head = nn.Sequential(
            nn.Linear(in_features=512, out_features=embedding_dims, dtype=torch.float32),
        )
        self.projection = nn.Sequential(
            nn.Linear(embedding_dims, embedding_dims),
            nn.ReLU(),
            nn.Linear(embedding_dims, embedding_dims)
        )
        
        
    def forward(self, x, mode):
        # training mode - compute loss on projection embeddings
        if mode == "train":
            embeddings = self.model.encode_image(x[0]).to(device)
            video_embeddings = PositionalEncoding(embeddings).to(device)
            video_embeddings = self.head(video_embeddings)
            projection_x = self.projection(video_embeddings)
            return projection_x
        # eval mode - evaluate on network embeddings
        elif mode == "eval": 
            embeddings = self.model.encode_image(x[0]).to(device)
            video_embeddings = PositionalEncoding(embeddings).to(device)
            video_embeddings = self.head(video_embeddings)
            return video_embeddings
        
