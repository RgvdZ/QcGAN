import os
import torch
import math
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

def calculate_psnr(img1, img2):
    """
    Calculate PSNR (Peak Signal-to-Noise Ratio).
    Expects tensors in the range [-1, 1].
    """
    # Denormalize from [-1, 1] to [0, 1]
    img1 = (img1 + 1.0) / 2.0
    img2 = (img2 + 1.0) / 2.0

    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(1.0 / math.sqrt(mse))

def save_checkpoint(state, filename="checkpoint.pth.tar"):
    torch.save(state, filename)

def train(generator, discriminator, train_loader, val_loader, num_epochs, device, save_dir="checkpoints"):
    os.makedirs(save_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(save_dir, 'logs'))

    # Optimizers: Adam, lr=0.0001
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.0001, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.0001, betas=(0.5, 0.999))

    # Learning rate schedule: Constant for 150 epochs, linear decay to 0 for next 150
    def lr_lambda(epoch):
        if epoch < 150:
            return 1.0
        else:
            return max(0.0, 1.0 - (epoch - 150) / 150.0)

    scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=lr_lambda)
    scheduler_D = torch.optim.lr_scheduler.LambdaLR(optimizer_D, lr_lambda=lr_lambda)

    perceptual_loss = PerceptualLoss().to(device)
    lambda_x = 100  # Content loss weight

    best_psnr = 0.0

    print("Starting training...")
    for epoch in range(num_epochs):
        generator.train()
        discriminator.train()

        epoch_g_loss = 0
        epoch_d_loss = 0

        for i, (blur_imgs, sharp_imgs) in enumerate(train_loader):
            blur_imgs = blur_imgs.to(device)
            sharp_imgs = sharp_imgs.to(device)

            # =======================
            #  1. Train Discriminator
            # =======================
            optimizer_D.zero_grad()

            fake_sharp_imgs = generator(blur_imgs)

            # Discriminator expects (Blur, Sharp) paired patches
            real_input = torch.cat((blur_imgs, sharp_imgs), dim=1)
            fake_input = torch.cat((blur_imgs, fake_sharp_imgs.detach()), dim=1)

            pred_real = discriminator(real_input)
            pred_fake = discriminator(fake_input)

            # Hinge Loss for Discriminator
            loss_D_real = torch.nn.ReLU()(1.0 - pred_real).mean()
            loss_D_fake = torch.nn.ReLU()(1.0 + pred_fake).mean()
            loss_D = loss_D_real + loss_D_fake

            loss_D.backward()
            optimizer_D.step()

            # =======================
            #  2. Train Generator
            # =======================
            optimizer_G.zero_grad()

            fake_input_for_G = torch.cat((blur_imgs, fake_sharp_imgs), dim=1)
            pred_fake_for_G = discriminator(fake_input_for_G)

            # Hinge loss for Generator (Adversarial)
            loss_G_adv = -pred_fake_for_G.mean()

            # Content Loss (Perceptual VGG Loss)
            loss_G_content = perceptual_loss(fake_sharp_imgs, sharp_imgs)

            # Total Generator Loss
            loss_G = loss_G_adv + (lambda_x * loss_G_content)

            loss_G.backward()
            optimizer_G.step()

            epoch_g_loss += loss_G.item()
            epoch_d_loss += loss_D.item()

            if i % 100 == 0:
                print(f"[Epoch {epoch}/{num_epochs}] [Batch {i}/{len(train_loader)}] "
                      f"[D loss: {loss_D.item():.4f}] [G loss: {loss_G.item():.4f}]")

        # Step schedulers at the end of epoch
        scheduler_G.step()
        scheduler_D.step()

        # Log training losses
        writer.add_scalar('Loss/Discriminator', epoch_d_loss / len(train_loader), epoch)
        writer.add_scalar('Loss/Generator', epoch_g_loss / len(train_loader), epoch)

        # =======================
        #  3. Validation & PSNR
        # =======================
        generator.eval()
        val_psnr = 0.0

        with torch.no_grad():
            for val_blur, val_sharp in val_loader:
                val_blur = val_blur.to(device)
                val_sharp = val_sharp.to(device)

                val_fake = generator(val_blur)

                # Calculate PSNR for this batch
                for b in range(val_blur.size(0)):
                    val_psnr += calculate_psnr(val_fake[b], val_sharp[b])

        avg_psnr = val_psnr / len(val_loader.dataset)
        writer.add_scalar('Metric/Validation_PSNR', avg_psnr, epoch)

        print(f"--> Epoch {epoch} Validation PSNR: {avg_psnr:.2f} dB")

        # =======================
        #  4. Checkpointing
        # =======================
        is_best = avg_psnr > best_psnr
        best_psnr = max(avg_psnr, best_psnr)

        checkpoint_state = {
            'epoch': epoch + 1,
            'generator_state_dict': generator.state_dict(),
            'discriminator_state_dict': discriminator.state_dict(),
            'optimizer_G': optimizer_G.state_dict(),
            'optimizer_D': optimizer_D.state_dict(),
            'best_psnr': best_psnr
        }

        # Save standard checkpoint
        save_checkpoint(checkpoint_state, filename=os.path.join(save_dir, "checkpoint_latest.pth.tar"))

        # Save best model
        if is_best:
            print(f"*** New best PSNR reached: {best_psnr:.2f} dB. Saving model... ***")
            save_checkpoint(checkpoint_state, filename=os.path.join(save_dir, "fmd_cgan_best.pth.tar"))

    writer.close()
    print("Training completely finished!")
