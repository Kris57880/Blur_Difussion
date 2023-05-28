import numpy as np
import torch
import sys
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision
import os
from diffusers import DDPMScheduler, UNet2DModel
from matplotlib import pyplot as plt
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm
import csv

device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)
print('Using device:', device)

image_size = 256
batch_size = 2
num_workers = 4
n_epochs = 200
lr = 1e-5
Dataset_dir = 'Dataset'
model_dir = 'DDDAAA_per/saved_model'
result_dir = 'DDDAAA_per/result'
log_file = 'DDDAAA_per/training_log.csv'

preprocess = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

class GOPRO(Dataset):
    def __init__(self, mode='train', transform=preprocess):
        assert mode == 'train' or mode == 'test'
        self.mode = mode
        # self.args = args
        self.transform = transform
        self.dirs = []
        self.dirs_gt = []
        self.index = 0
        self.d = 0

        for d1 in os.listdir(os.path.join('./Dataset', mode)):
            for d3 in os.listdir(os.path.join('./Dataset', mode, d1, 'blur_gamma')):
                if d3.endswith('.png'):
                    self.dirs.append(os.path.join(
                        './Dataset', mode, d1, 'blur_gamma', d3))
                    self.dirs_gt.append(os.path.join(
                        './Dataset', mode, d1, 'sharp', d3))

    def __len__(self):
        return len(self.dirs)

    def __getitem__(self, index):
        self.cur_dirs = self.dirs[index]
        self.cur_dirs_gt = self.dirs_gt[index]

        # image_seq = []
        # image_seq_gt = []

        im = self.transform(Image.open(self.cur_dirs)).reshape(
            (3, image_size, image_size))
        # image_seq.append(im)
        # image_seq = torch.Tensor(np.concatenate(image_seq, axis=0))
        im_gt = self.transform(Image.open(self.cur_dirs_gt)).reshape(
            (3, image_size, image_size))
        return im, im_gt

class ClassConditionedUnet(nn.Module):
    def __init__(self):
        super().__init__()
        
        #image embedding  3->1 channel
        # self.img_emb = nn.Conv2d(3,1, kernel_size=(1,1))

        # Self.model is an unconditional UNet with extra input channels to accept the conditioning information (the class embedding)
        self.model = UNet2DModel(
            sample_size=image_size,           # the target image resolution
            in_channels=3 + 3,  # Additional input channels for class cond.
            out_channels=3,           # the number of output channels
            layers_per_block=2,       # how many ResNet layers to use per UNet block
            block_out_channels=(128, 128, 256, 256, 512, 512),
            down_block_types=(
                "DownBlock2D",
                "DownBlock2D",
                "DownBlock2D",        # a regular ResNet downsampling block
                "AttnDownBlock2D",
                "AttnDownBlock2D",
                "AttnDownBlock2D",
            ),
            up_block_types=(
                "AttnUpBlock2D",
                "AttnUpBlock2D",
                "AttnUpBlock2D",
                "UpBlock2D",      # a ResNet upsampling block with spatial self-attention
                "UpBlock2D",          # a regular ResNet upsampling block
                "UpBlock2D",          
            ),
        )

    # Our forward method now takes the class labels as an additional argument
    def forward(self, x, t, blur_labels):
        bs, ch, w, h = x.shape
        # (bs, 3 + 1, 256, 256)
        # img_cond = self.img_emb(blur_labels) 
        
         # x is shape (bs, 1, 28, 28) and class_cond is now (bs, 4, 28, 28)
        net_input = torch.cat((x, blur_labels), axis=1)
        # class conditioning in right shape to add as additional input channels
        # class_cond = self.class_emb(class_labels) # Map to embedding dinemsion
        # class_cond = class_cond.view(bs, class_cond.shape[1], 1, 1).expand(bs, class_cond.shape[1], w, h)
        # x is shape (bs, 1, 28, 28) and class_cond is now (bs, 4, 28, 28)
        # Feed this to the unet alongside the timestep and return the prediction
        return self.model(net_input, t).sample  # (bs, 3 + 1, 256, 256)
if __name__ == '__main__':
    train_dataset = GOPRO(mode='train')
    # test_dataset = GOPRO(mode='test')
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size,
            num_workers=num_workers, shuffle=True, drop_last=True, pin_memory=True)

    # print(f'train data num: {len(train_dataset)} test data num: {len(test_dataset)}')
    print(f'image size: {image_size} batch size: {batch_size} num workers: {num_workers}')

    noise_scheduler = DDPMScheduler(
        num_train_timesteps=1000, beta_schedule='linear')
    alphas = noise_scheduler.alphas_cumprod
    snr = (alphas / (1-alphas))
    model_files = [f for f in os.listdir(model_dir) if os.path.isfile(os.path.join(model_dir, f)) and f.startswith('epoch_')]
    net = ClassConditionedUnet().to(device)
    # uncomment if want using multiple GPUs
    #net = nn.DataParallel(ClassConditionedUnet()).to(device)

    opt = torch.optim.Adam(net.parameters(), lr=lr)
    start_epoch = 0 

    if model_files: 
        latest_model = max([os.path.join(model_dir, f) for f in model_files], key=os.path.getctime)
        net.load_state_dict(torch.load(latest_model))
        start_epoch = int(latest_model[:-3].split('/')[-1].split('_')[1])
        print(f'resume from epoch {start_epoch}')

    # create the log file if it doesn't exist
    if not os.path.exists(log_file):
        with open(log_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Epoch', 'Loss'])

    # Our network
    orig_stdout = sys.stdout
    model_description = open(model_dir+"/model_description.txt",'w+')
    sys.stdout = model_description
    print(net)
    sys.stdout = orig_stdout
    model_description.close()
    # Our loss finction
    loss_fn = nn.MSELoss()
    # The optimizer
    print('number of params: ', sum([p.numel() for p in net.parameters()]))

    # Keeping a record of the losses for later viewing
    epoch_losses = []

    net.train()

    min_loss = float('inf')
    # The training loop
    for epoch in range(start_epoch+1, n_epochs+1):
        iter_losses = []
        losses_w = []
        for blur_img, gt in tqdm(train_dataloader):
            # Get some data and prepare the corrupted version
            blur_img = blur_img.to(device) # Normalize the data to [-1, 1]
            gt = gt.to(device)# Normalize the data to [-1, 1]
            noise = torch.randn_like(gt)
            # print(torch.min(blur_img))
            timesteps = torch.randint(0, 999, (gt.shape[0],)).long().to(device)
            noisy_x = noise_scheduler.add_noise(gt, noise, timesteps)

            # Get the model prediction
            # Note that we pass in the blur_img as condition
            pred = net(noisy_x, timesteps, blur_img)

            # Calculate the loss
            loss = loss_fn(pred, noise)  # How close is the output to the noise
            snr1 = snr[timesteps].to(device)
            # print(snr)
            # temp = ((1 + snr1)**0.5)
            # snr1 = abs(snr(pred, noise).to(device))
            # print('snr:',snr)
            loss_w = (loss / ((1 + snr1)**1)).sum()
            # Backprop and update the params:
            opt.zero_grad()
            loss_w.backward()
            opt.step()

            # Store the loss for later
            losses_w.append(loss_w.item())
            iter_losses.append(loss.item())

        # Print our the average of the last 100 loss values to get an idea of progress:
        avg_loss = sum(iter_losses)/len(iter_losses)
        epoch_losses.append(avg_loss)
        print(f'Finished epoch {epoch}. Loss values: {avg_loss:05f} / {sum(losses_w)/len(losses_w):05f}')
        with open(log_file, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([epoch, avg_loss])
        if epoch %10 == 0:
            torch.save(net.state_dict(), model_dir+f'/epoch_{epoch}.pt')

    torch.save(net.state_dict(), model_dir+f'/final.pt')
    # View the loss curve
    plt.plot(epoch_losses)
    plt.title('Loss curve')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.savefig(result_dir+'/loss_curve.png')
    plt.show()
