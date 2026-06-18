 #Step -  3: TRAININ THE DATA

optimizer = torch.optim.Adam(model.parameters(),lr = 0.001)
loss_fn = nn.CrossEntropyLoss()

num_epochs  = 50000
batch_size = 32

losses_train = []
losses_val = []

print("Training....\n")

for epoch in range(num_epochs):
  #Training
  #Sample a random batch
  indices = torch.randperm(len(Xtr))[:batch_size]
  Xbatch,Ybatch = Xtr[indices],Ytr[indices]
  #Forward Pass
  logits = model(Xbatch)
  loss = loss_fn(logits,Ybatch)
  #Backward pass
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()
  losses_train.append(loss.item())

  #Validation Every 1000 epochs
  if epoch%1000 ==0:
    with torch.no_grad():
        # Validation loss (full dataset)
            logits_val = model(Xval)
            loss_val = loss_fn(logits_val, Yval)
            losses_val.append(loss_val.item())

            print(f"Epoch {epoch:5d} | Train loss: {loss.item():.4f} | Val loss: {loss_val.item():.4f}")




