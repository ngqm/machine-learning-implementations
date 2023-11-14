import torch 
from ddpm import DiffusionModule
from scheduler import DDPMScheduler
import numpy as np
from dataset import tensor_to_pil_image, pil_image_to_tensor
from torchvision.transforms.functional import to_pil_image, to_tensor
from PIL import Image

# load trained model
path = 'image_diffusion_todo/results/diffusion-07-23-201252/last.ckpt'
dic = torch.load(path, map_location='cpu')
hparams = dic['hparams']
state_dict = dic['state_dict']

# create a new scheduler instance (to get newly implemented methods)
scheduler = DDPMScheduler(
        1000,
        beta_1=1e-4,
        beta_T=0.02,
        mode="linear",
)

# create a DDPM
ddpm = DiffusionModule(None, None)
ddpm.network = hparams['network']
ddpm.var_scheduler = scheduler
ddpm.load_state_dict(state_dict)
ddpm.eval()
ddpm.to('cuda:9')

# create batches of images from a path
sample_path = 'image_diffusion_todo/samples/diffusion-07-23-201252'
save_path = 'image_diffusion_todo/repaint/diffusion-07-23-201252'
masked_path = 'image_diffusion_todo/masked/diffusion-07-23-201252'
batch_size = 32
total_num_samples = 500
num_batches = int(np.ceil(total_num_samples / batch_size))
for i in range(num_batches):
    # load images of a batch into a tensor
    sidx = i * batch_size
    eidx = min(sidx + batch_size, total_num_samples)
    num_images = eidx - sidx
    images = torch.zeros([num_images, 3, 64, 64])
    pil_images = []
    for j in range(sidx, eidx):
        img = Image.open(f'{sample_path}/{j}.png')
        pil_images.append(img)
    images = pil_image_to_tensor(pil_images)
    # create uniformly random 32x32 rectangular masks
    masks = torch.ones([num_images, 3, 64, 64])
    for j in range(num_images):
        x = np.random.randint(0, 32)
        y = np.random.randint(0, 32)
        masks[j, :, x:x+32, y:y+32] = 0
    # save the images with masks
    masked_images = images * masks
    pil_images = tensor_to_pil_image(masked_images)
    for j, pil_image in zip(range(sidx, eidx), pil_images):
        pil_image.save(f'{masked_path}/{j}.png')
        print(f"Saved the {j}-th masked image.")
    
    # sample from the model
    samples = ddpm.repaint(num_images, masks, original_samples=images)
    pil_images = tensor_to_pil_image(samples)
    # save tbe sampled images
    for j, pil_image in zip(range(sidx, eidx), pil_images):
        pil_image.save(f'{save_path}/{j}.png')
        print(f"Saved the {j}-th inpainted image.")
