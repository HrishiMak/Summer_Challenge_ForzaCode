import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn
from einops import repeat

class ClassificationNet(nn.Module):
    '''
    Defines a neural network for classification with three fully connected layers

    Args:
        input_size: Size of input features
    '''
    def _init_(self,input_size):
        super()._init_()
        self.input_size=input_size
        self.fc1=nn.Sequential(nn.Linear(input_size,16),
                               nn.LeakyReLU(),
                               nn.Linear(16,8),
                               nn.LeakyReLU(),
                               nn.Linear(8,1))
        self.sigmoid=nn.Sigmoid()

    def forward(self,x):
        out=self.fc1(x)
        return self.sigmoid(out)
      
class SiameseNet(nn.Module):
    '''
    Implements the Siamese Network architecture.

    Args:
        embedding_net: a separate neural network module that will be used to compute the embeddings for input samples.
    '''
    def __init__(self, embedding_net):
        super(SiameseNet, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, x1, x2):
        '''
        Defines the forward pass of the SiameseNet. Takes two input samples, x1 and x2, and passes them through the embedding_net 
        to obtain their respective embeddings, output1 and output2. 

        Args:
            x1, x2: Tensor representations of the respective images

        Returns:
            output1, output2: Embeddings of the input tensors
        '''
        output1 = self.embedding_net(x1)
        output2 = self.embedding_net(x2)
        return output1, output2

    def get_embedding(self, x):
        '''
        Convenience method that allows obtaining the embeddings for a single input sample.

        Args:
            x: Input tensor

        Returns:
            self.embedding_net(x): Embedding of input tensor
        '''
        return self.embedding_net(x)


def pair(t):
    '''
    Checks the input to be a tuple or not and pairs into a tuple if not.
    '''
    return t if isinstance(t, tuple) else (t, t)

class PreNorm(nn.Module):
    '''
    Performs pre-normalization on the input before applying a specified function to it.

    Args:
        dim (int): Size of the last dimension of the input tensor.
        fn : function to be applied to the normalized input.
    '''
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        '''
        Defines the forward pass of the PreNorm module
      
        Args:
            x: Input tensor
            **kwargs: Used to capture any additional keyword arguments that are passed to the function
      
        Returns:
            The application of normalized input to the particular function.
        '''
        x = x.transpose(1, -1)
        x = self.norm(x)
        x = x.transpose(-1, 1)
        return self.fn(x, **kwargs)

class FeedForward(nn.Module):
    '''
    Implements a feed-forward network with two convolutional layers and GELU activation

    Args:
    dim(tuple): Input dimension
    hidden_dim(tuple): Dimension of the hidden layer
    dropout(float): Dropout probability with default value=0.
    '''
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, 1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(hidden_dim, dim, 1),
            nn.Dropout(dropout)
        )
  
    def forward(self, x):
        '''
        Forward pass of the module.
      
        Args:
            x: Input tensor
        '''
        return self.net(x)

class Attention(nn.Module):
    '''
    Implements the self-attention mechanism commonly used in transformer-based models.

    Args:
        dim(int): Input dimension
        heads(int): Number of attention heads with default value = 4
        dim_heads(int): Dimension of each attention head
        dropout(float): Dropout probability with default value = 0.
    '''
    def __init__(self, dim, heads = 4, dim_head = 32, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads  #total dimensionality of all the attention heads combined.
        project_out = not (heads == 1 and dim_head == dim) #determines whether projection of the output is required or if an identity mapping can be used.

        self.heads = heads
        self.head_channels = dim // self.heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -2)
        self.to_keys = nn.Conv2d(dim, inner_dim, 1)
        self.to_queries = nn.Conv2d(dim, inner_dim, 1)
        self.to_values = nn.Conv2d(dim, inner_dim, 1)
        self.unifyheads = nn.Sequential(                          #projecting the concatenated outputs of the attention heads back to the original dimension dim.
            nn.Conv2d(inner_dim, dim, 1),
            nn.Dropout(dropout)) if project_out else nn.Identity()

    def forward(self, x):
        '''
        Forward pass of Attention module. 

        Args:
            x: Input tensor
        '''
        b, _, h, w = x.shape
        #to_keys, to_queries, and to_values layers are applied to the input tensor to obtain key (k), query (q), and value (v) tensors, respectively. 
        k = self.to_keys(x).view(b, self.heads, self.head_channels, -1) 
        q = self.to_queries(x).view(b, self.heads, self.head_channels, -1)
        v = self.to_values(x).view(b, self.heads, self.head_channels, -1)

        dots = k.transpose(-2, -1) @ q

        attn = self.attend(dots)

        out = torch.matmul(v, attn)
        out = out.view(b, -1, h, w)
        return self.unifyheads(out)

class Transformer(nn.Module):
    '''
    A stack of transformer layers.
    '''
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth): #A loop is used to add depth number of transformer layers to the layers list.
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers: #self-attention block and feed-forward block are extracted from the layer and applied to input tensor x.
            x = attn(x) + x
            x = ff(x) + x
        return x

class ViT(nn.Module):
    '''
    Implementation of the Vision Transformer (ViT) model, which combines the transformer architecture with a convolutional neural network (CNN) backbone for image classification.

    Args:
        image_size(tuple): A tuple specifying the height and width of the input image.
        patch_size(tuple): A tuple specifying the height and width of each patch that the image is divided into.
        num_classes(int): The number of output classes.
        dim: The dimensionality of the transformer hidden state.
        depth(int): The number of transformer layers to stack.
        heads(int): The number of attention heads in each transformer layer.
        mlp_dim: The dimensionality of the feed-forward network hidden layer.
        pool(string): The pooling type, which can be 'cls' (cls token) or 'mean' (mean pooling).
        channels(int): The number of input image channels (default is 3 for RGB images).
        dim_head(int): The dimensionality of each attention head.
        dropout(float): The dropout probability for both the transformer and patch embeddings.
        emb_dropout(float): The dropout probability for the patch embeddings only.
    '''
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 32, dropout = 0., emb_dropout = 0.):
        super(ViT,self).__init__()
        image_height, image_width = image_size
        patch_height, patch_width = patch_size

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            nn.Conv2d(channels, dim, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(dim, dim, kernel_size=patch_size, stride=patch_size)
        )

        
        reduced_height = image_height // patch_height
        reduced_width = image_width // patch_width
        shape = (reduced_height, reduced_width)
        dim += 1
        self.pos_embedding = nn.Parameter(torch.randn(dim, *shape))

        self.cls_token = nn.Parameter(torch.randn(1, 1, *shape))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()
        self.mlp_head = nn.Sequential(nn.Flatten(),
                                      nn.Linear(reduced_height*reduced_width, num_classes))
        self.flatten=nn.Flatten()
        self.reset_parameters()

    def forward(self, img):
        '''
        Forward pass operation.
        '''
        x = self.to_patch_embedding(img)

        b, n, h, w = x.shape

        cls_tokens = repeat(self.cls_token, '() n h w -> b n h w', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        
        x += self.pos_embedding[:, :(n + 1)]
        # x += self.pos_embedding
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)

        x=self.flatten(x)
        #return self.mlp_head(x)
        return x

    def reset_parameters(self):
        '''
        Initialize the weights and biases of the model.
        '''
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1.)
                nn.init.zeros_(m.bias)
    
    def separate_parameters(self):
        '''
        Separate the parameters of the model into weight decay and no weight decay groups.
        '''
        parameters_decay = set()
        parameters_no_decay = set()
        modules_weight_decay = (nn.Linear, nn.Conv2d)
        modules_no_weight_decay = (nn.LayerNorm,)

        for m_name, m in self.named_modules():
            for param_name, param in m.named_parameters():
                full_param_name = f"{m_name}.{param_name}" if m_name else param_name

                if isinstance(m, modules_no_weight_decay):
                    parameters_no_decay.add(full_param_name)
                elif param_name.endswith("bias"):
                    parameters_no_decay.add(full_param_name)
                elif isinstance(m, modules_weight_decay):
                    parameters_decay.add(full_param_name)
