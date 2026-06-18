
"""
KEY INSIGHT: Deep networks are fragile without batch normalization.

Why?
- If weights are initialized too big: activations saturate (all 0 or 1)
- If weights are too small: gradients vanish (too close to 0)
- If activations aren't normalized: each layer sees changing input distributions

Batch Normalization fixes this by:
1. Normalizing inputs to each layer to mean=0, std=1
2. Learning scale (gamma) and shift (beta) parameters
3. Stabilizes training, allows deeper networks

The math:
  normalized = (x - batch_mean) / sqrt(batch_var + epsilon)
  output = gamma * normalized + beta

PyTorch handles this automatically with nn.BatchNorm1d()
"""
#Step - 2: BUILD NETWORK WITH BATCH NORMALIZATION
class MakeMoreNet(nn.Module):
  def __init__(self ,vocab_size,embedding_dim,hidden_dims,block_size):
        super().__init__()
## Embedding layer: convert character indices to dense vectors
        self.embedding = nn.Embedding(vocab_size,embedding_dim)

        #Build hidden layers with batch norm.
        layers = []
        input_dim = embedding_dim*block_size #flatten concatenated embedding

        for hidden_dim in hidden_dims:
          #Linear Layer
          layers.append(nn.Linear(input_dim,hidden_dim))
          #Batch Normalization
          layers.append(nn.BatchNorm1d(hidden_dim))
          #Activation tanh squashes logits to [-1,1]
          layers.append(nn.Tanh())
          input_dim = hidden_dim
        #Output layer
        layers.append(nn.Linear(input_dim , vocab_size))
        self.layers =  nn.Sequential(*layers)

  def forward(self,x):
   #x shape : (batch_size,black_size)
   #Embed and flatten
   embedded = self.embedding(x)
   flattened = embedded.view(embedded.shape[0],-1)
   #Pass through layers
   logits = self.layers(flattened)
   return(logits)

vocab_size = len(chars)
embedding_dim = 10
hidden_dims = [200,200]
block_size = 3
model = MakeMoreNet(vocab_size,embedding_dim,hidden_dims,block_size)

print("Network architecture:")
print(f"  Input: {block_size} characters")
print(f"  Embedding: {vocab_size} chars → {embedding_dim}-dim vectors")
print(f"  Flatten: {block_size * embedding_dim} → {hidden_dims[0]}")
print(f"  Hidden 1: {hidden_dims[0]} (with batch norm + tanh)")
print(f"  Hidden 2: {hidden_dims[1]} (with batch norm + tanh)")
print(f"  Output: {vocab_size} logits\n")

total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params}\n")



