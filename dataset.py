#Step - 1 : Build the dataset and read the file
words = open('names.txt','r').read().splitlines()
chars = ['.'] + sorted(list(set(''.join(words))))
stoi = {s: i for i, s in enumerate(chars)}
itos = {i: s for i, s in enumerate(chars)}
# Buildin dataset
block_size = 3  # context: look at 3 characters to predict the 4th
X, Y = [], []

for w in words :
  context = [0]*block_size
  for ch in w + '.':
    ix = stoi[ch]
    X.append(context)
    Y.append(ix)
    context = context[1:] + [ix]

X = torch.tensor(X)
Y = torch.tensor(Y)

#Split into train/val/test
n1 = int(0.8 * len(X))
n2 = int(0.9 * len(X))

Xtr, Ytr = X[:n1], Y[:n1]
Xval, Yval = X[n1:n2], Y[n1:n2]
Xtest, Ytest = X[n2:], Y[n2:]

print(f"Training Examples: {len(Xtr)}")
print(f"Validation Examples: {len(Xval)}")
print(f"Test Examples: {len(Xtest)}\n")
print(f"Total Examples:", len(Xtr) + len(Xval) + len(Xtest))





