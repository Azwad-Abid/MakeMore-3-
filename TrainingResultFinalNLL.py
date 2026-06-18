"""

Epoch     0 | Train loss: 3.4296 | Val loss: 3.3673
Epoch  1000 | Train loss: 2.7203 | Val loss: 2.5375
Epoch  2000 | Train loss: 2.3560 | Val loss: 2.5078
Epoch  3000 | Train loss: 2.2646 | Val loss: 2.4692
Epoch  4000 | Train loss: 2.2161 | Val loss: 2.4390
Epoch  5000 | Train loss: 1.9818 | Val loss: 2.4690
Epoch  6000 | Train loss: 2.0150 | Val loss: 2.4390
Epoch  7000 | Train loss: 2.2837 | Val loss: 2.4166
Epoch  8000 | Train loss: 2.3634 | Val loss: 2.4178
Epoch  9000 | Train loss: 1.9675 | Val loss: 2.4077
Epoch 10000 | Train loss: 2.2092 | Val loss: 2.3965
Epoch 11000 | Train loss: 2.0297 | Val loss: 2.4081
Epoch 12000 | Train loss: 1.9880 | Val loss: 2.4018
Epoch 13000 | Train loss: 2.0882 | Val loss: 2.3877
Epoch 14000 | Train loss: 1.9082 | Val loss: 2.3989
Epoch 15000 | Train loss: 2.5134 | Val loss: 2.3775
Epoch 16000 | Train loss: 2.2974 | Val loss: 2.3826
Epoch 17000 | Train loss: 1.8506 | Val loss: 2.3939
Epoch 18000 | Train loss: 2.2160 | Val loss: 2.3864
Epoch 19000 | Train loss: 2.3570 | Val loss: 2.3680
Epoch 20000 | Train loss: 1.8856 | Val loss: 2.3809
Epoch 21000 | Train loss: 1.9133 | Val loss: 2.3625
Epoch 22000 | Train loss: 2.0974 | Val loss: 2.3732
Epoch 23000 | Train loss: 1.8365 | Val loss: 2.3645
Epoch 24000 | Train loss: 1.7553 | Val loss: 2.3669
Epoch 25000 | Train loss: 1.5728 | Val loss: 2.3625
Epoch 26000 | Train loss: 2.2393 | Val loss: 2.3841
Epoch 27000 | Train loss: 2.0820 | Val loss: 2.3601
Epoch 28000 | Train loss: 1.6168 | Val loss: 2.3511
Epoch 29000 | Train loss: 1.7897 | Val loss: 2.3782
Epoch 30000 | Train loss: 2.0226 | Val loss: 2.3784
Epoch 31000 | Train loss: 1.8810 | Val loss: 2.3572
Epoch 32000 | Train loss: 1.9334 | Val loss: 2.3633
Epoch 33000 | Train loss: 2.0420 | Val loss: 2.3629
Epoch 34000 | Train loss: 2.1852 | Val loss: 2.3751
Epoch 35000 | Train loss: 1.8522 | Val loss: 2.3581
Epoch 36000 | Train loss: 2.3374 | Val loss: 2.3555
Epoch 37000 | Train loss: 2.1020 | Val loss: 2.3517
Epoch 38000 | Train loss: 2.0278 | Val loss: 2.3539
Epoch 39000 | Train loss: 2.1380 | Val loss: 2.3545
Epoch 40000 | Train loss: 1.9472 | Val loss: 2.3591
Epoch 41000 | Train loss: 1.8865 | Val loss: 2.3663
Epoch 42000 | Train loss: 2.1147 | Val loss: 2.3640
Epoch 43000 | Train loss: 2.2520 | Val loss: 2.3566
Epoch 44000 | Train loss: 2.0241 | Val loss: 2.3567
Epoch 45000 | Train loss: 2.1776 | Val loss: 2.3474
Epoch 46000 | Train loss: 1.8535 | Val loss: 2.3511
Epoch 47000 | Train loss: 1.7249 | Val loss: 2.3453
Epoch 48000 | Train loss: 1.7413 | Val loss: 2.3674
Epoch 49000 | Train loss: 2.3489 | Val loss: 2.3355
"""
with torch.no_grad():
    logits_test = model(Xtest)
    loss_test = loss_fn(logits_test, Ytest)

print(f"\nFinal test loss: {loss_test.item():.4f}\n")

"""FINAL LOSS = 2.3746""" : D
