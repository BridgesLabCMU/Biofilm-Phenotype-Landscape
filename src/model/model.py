from .utils import *



def PositionalEncoding(embeddings, max_len=512):
    model_dims = embeddings.shape[1]


    two_i = torch.arange(0, model_dims, 2, dtype=torch.float32, device=device)

    div_term = 1 / (10000 ** (two_i / 512))
    position = torch.arange(0,max_len,dtype=torch.float, device=device).unsqueeze(1)

    encoding = torch.zeros(max_len, model_dims, device=device)
    encoding[:,0::2] = torch.sin(position * div_term)
    encoding[:,1::2] = torch.cos(position * div_term)
    image_embeddings = torch.add(embeddings, encoding[:embeddings.shape[0], :embeddings.shape[1]])
    
    return torch.mean(image_embeddings, axis=0)
        
class ViT(nn.Module):
    def __init__(self, model, num_frames=25, embedding_dims=512, num_classes=9):
        """
        Loads Vision Transformer from An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale.
        
        Arguments:
        num_frames - number of frames in input video
        embedding_dims - output embedding dimension 
        """
        super(ViT, self).__init__()
        # model.heads = nn.Sequential(
        #     nn.Linear(in_features=768, out_features=embedding_dims),
        # )
        self.model = model
        self.output = nn.Sequential(
            nn.Linear(embedding_dims, num_classes),
            nn.Softmax(-1)
        )
        self.projection = nn.Sequential(
            nn.Linear(embedding_dims, embedding_dims),
            nn.ReLU(),
            nn.Linear(embedding_dims, embedding_dims)
        )
        
        
    def forward(self, x, mode):
        # x shape - num_frames, channels, height, width
        
        
        x = self.model.encode_image(x)
        # x = x.last_hidden_state
        x = PositionalEncoding(x)
        x = x.to(device)
        classes = self.output(x)
        if mode == "train":
            projection_x = self.projection(x)
            return projection_x
        elif mode == "eval":
            return x