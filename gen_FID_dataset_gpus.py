from Baseline_Large.diffusion_large import GOPRO, ClassConditionedUnet
import torch
from tqdm.auto import tqdm
from matplotlib import pyplot as plt
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torch import nn
from diffusers import DDIMScheduler,DDPMScheduler
import os
from collections import OrderedDict

os.environ['CUDA_VISIBLE_DEVICES'] = "1,2,3,4"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

guidance_loss_scale = 30
def color_loss(images, target_color):
    """Given a target color (R, G, B) return a loss for how far away on average
    the images' pixels are from that color. Defaults to a light teal: (0.1, 0.9, 0.5)"""
    target = (
        torch.tensor(target_color).to(images.device) 
    )  # Map target color to (-1, 1)
    target = target[
        None, :, None, None
    ]  # Get shape right to work with the images (b, c, h, w)
    error = torch.abs(
        images - target
    ).mean()  # Mean absolute difference between the image pixels and the target color
    return error

def main():
    print('Using device', device)
        
    torch.manual_seed(87)

    # load model
    root_dir = 'Baseline_Large'
    net = nn.DataParallel(ClassConditionedUnet(),device_ids=[0,1,2,3])
    # net = net.module
    net.load_state_dict(torch.load(root_dir+'/saved_model/best.pt'),False)
    
    # #uncomment to accomodate multi-gpu evaluate by single gpu's result
    # new_state_dict = OrderedDict()
    # for k, v in state_dict.items():
    #     name = k[7:]  # remove `module.`
    #     new_state_dict[name] = v
    # # load params
    # net.load_state_dict(new_state_dict,False)
    # net.to(device)
    
    # load data
    image_size = 256
    batch_size = 32
    num_workers = 4
    TEST_PATH = 'Dataset/test'
    test_dataset = GOPRO(mode='test')
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size,
                                    num_workers=num_workers, shuffle=False, drop_last=True, pin_memory=True)
        

    # inference
    x = torch.randn(batch_size, 3, image_size, image_size).to(device)
    # y = torch.tensor([[i]*8 for i in range(10)]).flatten
    # ().to(device)



    noise_scheduler = DDPMScheduler(
            num_train_timesteps=1000, beta_schedule='linear')

    net.eval()
    net.to(device)

    # Sampling loop
    to_pil = transforms.ToPILImage()
    img_count = 0
    # Loop through the sampling timesteps
    for blur, sharp in tqdm(test_dataloader):
        # if img_count > 10:
        #     break

        y = blur.to(device)   # Normalize the data to [-1, 1]
        gt = sharp
        for j, t in enumerate(noise_scheduler.timesteps):

            # Prepare model input
            
            model_input = noise_scheduler.scale_model_input(x, t)
            #print(model_input.shape, t.shape ,y.shape)

            # Get the prediction
            with torch.no_grad():
                t_ = t.expand((gt.shape[0],)).to(device)
                noise_pred = net(model_input, t_, y)

            # Set x.requires_grad to True
            x = x.detach().requires_grad_()

            # Get the predicted x0
            x0 = noise_scheduler.step(noise_pred, t, x).pred_original_sample

            # Calculate color loss
            loss = color_loss(x0,y) * guidance_loss_scale
            # if j % 100 == 0:
            #     print(j, "loss:", loss.item())
            # Get gradient
            cond_grad = -torch.autograd.grad(loss, x)[0]

            # Modify x based on this gradient
            x = x.detach() + cond_grad

            # Now step with scheduler
            scheduler_output = noise_scheduler.step(noise_pred, t, x)

            # Update x
            x = scheduler_output.prev_sample
            loss_fn = nn.MSELoss()
            #save pred to result folder
            if j == len(noise_scheduler.timesteps) - 1:
                for im_x, im_gt in zip(x, gt):
                    im_x = to_pil(im_x.cpu().clip(-1, 1) * 0.5 + 0.5)
                    im_x.save(root_dir+'/result/pred/%04d.png'% img_count, 'png')
                    im_gt = to_pil(im_gt.cpu().clip(-1, 1) * 0.5 + 0.5)
                    im_gt.save(root_dir+'/result/gt/%04d.png' % img_count, 'png')
                    img_count += 1


if __name__ == '__main__':
    main()