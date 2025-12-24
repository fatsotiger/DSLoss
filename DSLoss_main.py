
MyEnsembleNet = fusion_net()
MyEnsembleNet = MyEnsembleNet.to(device)
triplet_loss = ContrastLoss().to(device)

#---------------init----------------------
w = 0.6  # 初始：完全雾
L_prev = float('inf')
eta = 1
w_min = 0.2
I_neg = None

# --- Strat training --- #
iteration = 0
for epoch in range(train_epoch):
    # ......
    for batch_idx, (hazy, clean) in enumerate(train_loader):  # 迭代训练集加载器，获取输入图像对，包括有雾的输入图像hazy和清晰的目标图像clean。

        hazy = hazy.to(device)
        clean = clean.to(device)
        output = MyEnsembleNet(hazy)

        # -----------DSLoss------------------
        H_true = hazy - clean
        I_neg = (hazy.detach() - (1 - w) * H_true)
        I_neg_out = (output.detach() + w * H_true)
        trip_loss = w * triplet_loss(output, clean, I_neg ) + (1 - w) * triplet_loss(output, clean, I_neg_out)
        # -----------------------------

        smooth_loss_l1 = F.smooth_l1_loss(output, clean)
        perceptual_loss = loss_network(output, clean)
        # msssim_loss_ = -msssim_loss(output, clean, normalize=True)
        total_loss = smooth_loss_l1  + 0.1 * trip_loss

        MyEnsembleNet.zero_grad()
        total_loss.backward()
        G_optimizer.step()
    #----------------------------updata W-----------------------------------#
    H_true = hazy - clean
    L_curr = smooth_loss_l1.item()
    dL = L_prev - L_curr

    current_dw = 0.0

    if L_prev == float('inf'):
        L_prev = L_curr
        continue

    if dL > 0:
        dw = -eta * dL * w
        current_dw = dw
        print(w_min, w, dw, w + dw)
        w = max(w_min, w + dw)
    L_prev = L_curr
    # ---------------------------------------------------------------#

