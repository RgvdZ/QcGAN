import argparse
import torch
from torch.utils.data import DataLoader
from dataset import DeblurDataset
from model import Generator, Discriminator, PerceptualLoss
from train import train

def main():
    parser = argparse.ArgumentParser(description="FMD-cGAN PyTorch Implementation")
    parser.add_argument('--dataroot', type=str, required=True, help="Path to the dataset containing 'blur' and 'sharp' folders")
    parser.add_argument('--batch_size', type=int, default=1, help="Batch size kept at 1 which gives better results [cite: 282]")
    parser.add_argument('--epochs', type=int, default=300, help="Total epochs (150 steady + 150 linear decay) [cite: 279, 280]")
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help="Device to run on")

    args = parser.parse_args()

    print(f"Using device: {args.device}")

    # Initialize Dataset and DataLoader
    dataset = DeblurDataset(root_dir=args.dataroot, is_train=True)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    # Initialize Generator and Discriminator
    # We use ngf=64 as established in the ablation study for optimal efficiency/performance [cite: 307]
    generator = Generator(ngf=64).to(args.device)
    discriminator = Discriminator(in_channels=6).to(args.device)

    # Start Training
    train(
        generator=generator,
        discriminator=discriminator,
        dataloader=dataloader,
        num_epochs=args.epochs,
        device=args.device
    )

    # Save final model
    torch.save(generator.state_dict(), "fmd_cgan_generator.pth")
    print("Training complete. Model saved to fmd_cgan_generator.pth")

if __name__ == '__main__':
    main()
