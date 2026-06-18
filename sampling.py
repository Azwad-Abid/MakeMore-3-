"""STEP - 5 SAMPLING"""
print("Generating samples:\n")

for i in range(20):
    context = [0] * block_size  # start with '.'
    out = []

    while True:
        # Forward pass
        model.eval()
        with torch.no_grad():
      
             logits = model(torch.tensor([context]))

        # Get probabilities
        probs = torch.softmax(logits, dim=1)

        # Sample
        ix = torch.multinomial(probs, num_samples=1).item()

        out.append(itos[ix])

        # Update context
        context = context[1:] + [ix]

        if ix == 0:  # stop at '.'
            break

    print(''.join(out))

print()

