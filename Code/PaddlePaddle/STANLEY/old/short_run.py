
import sys 
sys.path.append(".")
import paddle 
from utils.visual import make_grid, save_image, tensor2img
import numpy as np
import os
import torch
import pytorch_fid_wrapper as pfw

seed = 1
im_sz = 32
sigma = 3e-2 #decreaseuntiltrainingisunstable
n_ch = 3
batch_size = 8**2

test_batch_size = 5000
# K = 100
n_f = 64#increaseuntilcomputeisexhausted
n_i = 10**5
save_fid=True

class Fid_calculator(object):

    def __init__(self, training_data):
        pfw.set_config(batch_size=200, device=torch.device('cuda'))
        training_data = training_data.repeat(1,3 if training_data.shape[1] == 1 else 1,1,1)
        print("precalculate FID distribution for training data...")
        self.real_m, self.real_s = pfw.get_stats(training_data)

    def fid(self, data): 
        data = data.repeat(1,3 if data.shape[1] == 1 else 1,1,1) 
        return pfw.fid(data, real_m=self.real_m, real_s=self.real_s)


class EnergyBasedImageShortRun(paddle.nn.Layer):
    def __init__(self, input_sz=32, input_nc=3, nef=64, **kwargs):
        super(EnergyBasedImageShortRun, self).__init__()
        negative_slope = 0.2
        self.conv = paddle.nn.Sequential(
            paddle.nn.Conv2D(input_nc, nef, 3, 1, 1),
            paddle.nn.LeakyReLU(negative_slope),
            paddle.nn.Conv2D(nef, nef*2, 4, 2, 1),
            paddle.nn.LeakyReLU(negative_slope),
            paddle.nn.Conv2D(nef*2, nef*4, 4, 2, 1),
            paddle.nn.LeakyReLU(negative_slope),
            paddle.nn.Conv2D(nef*4, nef*8, 4, 2, 1),
            paddle.nn.LeakyReLU(negative_slope),
            paddle.nn.Conv2D(nef * 8, 1, 4, 1, 0))
    def forward(self, x):
        """Standard forward"""
        return self.conv(x).squeeze()

def main(K=100):
    print("run step size %d" % K)
    f = EnergyBasedImageShortRun()

    normalize = paddle.vision.transforms.Normalize(mean=127.5, std=127.5, data_format='HWC')
    dataset = paddle.vision.datasets.Cifar10(mode='train', transform=normalize)
    train_data = np.stack([x[0] for x in dataset]).swapaxes(2,3).swapaxes(1,2)
    os.makedirs("output_dir/short_run_step_%d" % (K), exist_ok=True)
    np.save("output_dir/short_run_step_%d/gt.npy" % (K), train_data)
    fid_calc = Fid_calculator(torch.from_numpy(train_data))
    train_data = paddle.Tensor(train_data)
    print(train_data.shape)
    def sample_q(K = K, test=False):
        x_k = paddle.rand(shape=(test_batch_size if test else batch_size,n_ch,im_sz,im_sz)) * 2 - 1
        for k in range(K):
            x_k.stop_gradient = False
            grad = paddle.grad([f(x_k).sum()], [x_k], retain_graph=True)[0]
            x_k += grad + 1e-2 * paddle.randn(shape=x_k.shape)
            x_k = x_k.detach()
        return x_k.detach()

    optim = paddle.optimizer.Adam(parameters=f.parameters(), learning_rate = 1e-4)
    best_fid = 10000
    for i in range(n_i):
        p_d_i = paddle.randint(0, train_data.shape[0], [batch_size])
        x_p_d = train_data[p_d_i] + sigma * paddle.randn(shape=(batch_size,n_ch,im_sz,im_sz))
        x_q = sample_q()
        obs_energy, syn_energy = f(x_p_d).mean(), f(x_q).mean()
        L = obs_energy - syn_energy
        optim.clear_grad()
        (-L).backward()
        optim.step()
        if i % 1000 == 0:
            img = make_grid(paddle.clip(x_q, -1, 1), 8)
            img = tensor2img(img, (-1, 1))
            save_image(img, "output_dir/short_run_step_%d/epoch_%d.png" % (K, i))
            if save_fid: 
                x = np.concatenate([sample_q(test=True).numpy() for j in range(10)])
                fid = fid_calc.fid(torch.from_numpy(x))
                # np.save("output_dir/short_run_step_%d/epoch_%d_%.4f.npy" % (K, i, fid), x)
                best_fid = min(fid, best_fid)
            print('{:>6d}f(x_p_d) = {:>14.9f}f(x_q) = {:>14.9f} fid {:>14.9f} best {:>14.9f}'.format(i, float(obs_energy), float(syn_energy), float(fid), float(best_fid)))
        

if __name__ == '__main__':
    
    if len(sys.argv) > 2:
        print("change gpu")
        os.environ['CUDA_VISIBLE_DEVICES'] = sys.argv[2]
    main(K=int(sys.argv[1]) if len(sys.argv) > 1 else 100)
    
