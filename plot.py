#PART6: VISUALIZATIONS (Understanding network internals)
"""
KEY INSIGHT: Visualizations help debug deep networks!

We want to check:
1. Are activations distributed nicely? (not saturated)
2. Are gradients flowing properly? (not vanishing or exploding)
3. Is the update:data ratio reasonable? (gradients moving weights enough)

""""

print("Generating diagnostic visualizations...\n")

# Get a batch to analyze
indices = torch.randperm(len(Xtr))[:10000]
Xbatch, Ybatch = Xtr[indices], Ytr[indices]

# Forward pass to collect activations
with torch.no_grad():
    embedded = model.embedding(Xbatch)
    flattened = embedded.view(embedded.shape[0], -1)

    activations = []
    x = flattened
    for layer in model.layers:
        if isinstance(layer, nn.Linear):
            x = layer(x)
            activations.append(x)
        elif isinstance(layer, (nn.BatchNorm1d, nn.Tanh)):
            x = layer(x)
            if isinstance(layer, nn.Tanh):
                activations.append(x)




# Visualize activation statistics
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# Plot 1: Activation means
means = [a.mean(dim=0).mean().item() for a in activations]
axes[0, 0].plot(means)
axes[0, 0].set_title('Mean of activations by layer')
axes[0, 0].set_ylabel('Mean activation')
axes[0, 0].axhline(y=0, color='r', linestyle='--')

# Plot 2: Activation stds
stds = [a.std(dim=0).mean().item() for a in activations]
axes[0, 1].plot(stds)
axes[0, 1].set_title('Std of activations by layer')
axes[0, 1].set_ylabel('Activation std')

# Plot 3: Activation distribution (last layer)
axes[1, 0].hist(activations[-1].flatten().numpy(), bins=50)
axes[1, 0].set_title('Distribution of final layer activations')
axes[1, 0].set_xlabel('Activation value')

# Plot 4: Loss over time
axes[1, 1].plot(losses_train[::10], label='Train')  # plot every 10th
if losses_val:
    axes[1, 1].plot(range(0, len(losses_train), 1000), losses_val, label='Val', marker='o')
axes[1, 1].set_title('Loss over training')
axes[1, 1].set_ylabel('Loss')
axes[1, 1].set_xlabel('Epoch (/ 10)')
axes[1, 1].legend()

plt.tight_layout()
plt.savefig('makemore_part3_diagnostics.png', dpi=100)
print("Saved diagnostic plot to makemore_part3_diagnostics.png\n")
