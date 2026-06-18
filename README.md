 MAKEMORE PART 3 :
 
Welcome! This repository contains my full implementation and core conceptual takeaways from Part 3 of Andrej Karpathy's Makemore series.

While Parts 1 and 2 focused on constructing the foundational Multilayer Perceptron (MLP) architecture, Part 3 pivots completely into network diagnostics, initialization math, and activation dynamics. The goal of this phase isn't just to make a model that generates text, but to look under the hood and analyze exactly how tensors mutate, saturate, and explode as they flow forward and backward through a deep neural network.

🛠️ The Architecture Blueprint
This model is a character-level deep language model that consumes a fixed window of characters (block_size) to predict a categorical probability distribution over the next character.

The network is built dynamically from a hyperparameters list using PyTorch's nn.Sequential block.

The Forward Pass Pipeline
Here is how data flows through the layers during a single forward pass:

Plaintext
  [ Input Tensor ]          Shape: (Batch Size, 3)  --> e.g., [[5, 13, 13]] for 'emm'
         │
         ▼
 ┌───────────────┐
 │ nn.Embedding  │          Shape: (Batch Size, 3, 10) --> Maps indices to 10D vectors
 └───────────────┘
         │
         ▼
 ┌───────────────┐
 │ Tensor .view  │          Shape: (Batch Size, 30) --> Flattens/concatenates context
 └───────────────┘
         │
         ▼
 ┌───────────────┐
 │  nn.Linear    │          Shape: (Batch Size, 200) --> Weights (30x200) + Biases (200)
 └───────────────┘
         │
         ▼
 ┌───────────────┐
 │ nn.BatchNorm1d│          Shape: (Batch Size, 200) --> Centers and normalizes channels
 └───────────────┘
         │
         ▼
 ┌───────────────┐
 │   nn.Tanh     │          Shape: (Batch Size, 200) --> Non-linear activation squashing
 └───────────────┘
         │
         ▼
 ┌───────────────┐
 │ Output Linear │          Shape: (Batch Size, 27) --> Projects features to vocab logits
 └───────────────┘
         │
         ▼
  [ 27 Logits ]             Passed to Cross-Entropy Loss (Target shape: Batch Size)
🧠 Core Engineering Lessons & Diagnostic Deep Dives
1. The Tanh "Dead Neuron" Saturation Crisis
When hidden units are initialized with weights that are too large, the outputs of the linear layer (XW+b) are pushed wildly into high positive or negative regimes. When these values hit nn.Tanh(), they get clamped completely to -1.0 or 1.0.

The Math: The derivative of tanh(x) is 1−tanh 
2
 (x). When tanh(x)=±1, the gradient evaluates to exactly 0.0.

The Implication: During backpropagation, this zero acts as a gatekeeper that kills the gradient chain. The error cannot flow backwards through that neuron, preventing it from ever updating its weights. This creates a "dead neuron" that adds zero capacity to the network.

The Fix: We resolve this by properly scaling down initial weights (Kaiming/He Initialization) to ensure the pre-activations land on the active, sloping portion of the Tanh curve.

2. Batch Normalization Mechanics
To solve initialization instability deterministically, we introduced Batch Normalization (nn.BatchNorm1d) layers right before our activation layers.

BatchNorm enforces stability by calculating the mean (μ) and variance (σ 
2
 ) across the mini-batch for every single channel, standardizing the activations:

x
^
 = 
σ 
2
 +ϵ

​
 
x−μ
​
 
The layer then applies trainable scale (γ) and shift (β) parameters to give the network the flexibility to undo the normalization if it helps reduce loss. This keeps our activations optimally distributed throughout training, drastically reducing our sensitivity to weight initialization tuning.

3. The model.eval() Trap
One of the most tricky bugs encountered occurs when trying to generate text (inference) after training. Because BatchNorm requires computing batch metrics on the fly, passing a single context window (Batch Size = 1) causes a division-by-zero variance crash:

Plaintext
ValueError: Expected more than 1 value per channel when training, got input size torch.Size([1, 200])
The Lesson: We must explicitly switch the model into evaluation state using model.eval(). This flags the BatchNorm modules to stop looking at batch shapes and start utilizing the running historical averages of mean and variance that it saved during training.

💻 Code Architecture Highlight
Here is how the deep network configuration is built cleanly and systematically using a dynamic loop:

Python
# Configuration Parameters
vocab_size = len(chars)  # 27 tokens (a-z and '.')
embedding_dim = 10
hidden_dims = [200, 200]
block_size = 3

# Layer Stacking Routine
layers = []
input_dim = embedding_dim * block_size

for hidden_dim in hidden_dims:
    layers.append(nn.Linear(input_dim, hidden_dim))
    layers.append(nn.BatchNorm1d(hidden_dim))
    layers.append(nn.Tanh())
    input_dim = hidden_dim

# Append Final Output Projection Layer
layers.append(nn.Linear(input_dim, vocab_size))
self.layers = nn.Sequential(*layers)
Using this structured pipeline, the model aggregates exactly 52,897 trainable parameters, distributed across lookups, weight matrices, bias vectors, and normalization scalars.

📈 Optimization & Hyperparameters
Loss Function: PyTorch Cross-Entropy Loss

Learning Rate Strategy: Learning Rate Decay. We initialize training at 0.1 to aggressively progress down the loss surface, then decay by a factor of 10 to 0.01 in later epochs to fine-tune the parameters and comfortably lock into the global minimum.

Batch Size: 32 examples per step

🤝 Acknowledgments
Immense credit to Andrej Karpathy for the masterclass on neural network internals. This project completely reframes how I view deep network layers—not as opaque black boxes, but as deterministic highways of flowing numerical gradients.
