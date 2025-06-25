from .utils import *



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
    def __init__(self, model, num_frames=25, embedding_dims=128, num_classes=9):
        """
        Loads Vision Transformer from An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale.
        
        Arguments:
        num_frames - number of frames in input video
        embedding_dims - output embedding dimension 
        """
        super(ViT, self).__init__()

        self.model = model
        
        self.head = nn.Sequential(
            nn.Linear(in_features=512, out_features=embedding_dims, dtype=torch.half),
        )
        
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
        

        
        

        # x = x.last_hidden_state
        # x = PositionalEncoding(x)
        
        # classes = self.output(x)
        if mode == "train":
            embeddings = torch.empty(x.shape[0], 128, device=device)
            for i, image in enumerate(x):
                frame_embeddings = self.model.encode_image(image)
                frame_embeddings = self.head(frame_embeddings)
                video_embeddings = PositionalEncoding(frame_embeddings).to(device)
                projection_x = self.projection(video_embeddings)
                embeddings[i] = projection_x
            return embeddings
        elif mode == "eval":
            embeddings = torch.empty(x.shape[0], 128, device=device)
            for i, image in enumerate(x):
                frame_embeddings = self.model.encode_image(image)
                frame_embeddings = self.head(frame_embeddings)
                embeddings[i] = frame_embeddings
            x = PositionalEncoding(embeddings)
            x = x.to(device)
            return x
        