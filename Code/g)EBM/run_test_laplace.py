import torch as t
import torch.nn as nn

import torchvision as tv
import torchvision.transforms as tr

seed = 1
im_sz = 32
sigma = 3e-2
n_ch = 3
m = 8**2

K = 100
n_f = 64
n_i = 10**5

t.manual_seed(seed)

if t.cuda.is_available():
    t.cuda.manual_seed_all(seed)

print("==> Chosing device...")

device = t.device('cuda' if t.cuda.is_available() else 'cpu')
# device  = t.device('cpu')

class F(nn.Module):
    def __init__(self, n_c = n_ch, n_f = n_f, l = 0.2):
        super(F, self).__init__()
        self.f = nn.Sequential(
            nn.Conv2d(n_c, n_f, 3,1,1),
            nn.LeakyReLU(l),
            nn.Conv2d(n_f, n_f*2, 4,2,1),
            nn.LeakyReLU(l),
            nn.Conv2d(n_f*2, n_f*4, 4,2,1),
            nn.LeakyReLU(l),
            nn.Conv2d(n_f*4, n_f*8, 4,2,1),
            nn.LeakyReLU(l),
            nn.Conv2d(n_f*8, 1, 4,1,0))

    def forward(self, x):
        return self.f(x).squeeze()

f = F().to(device)

transform=tr.Compose([tr.Resize(im_sz),tr.ToTensor(),tr.Normalize((.5,.5,.5),(.5,.5,.5))])
p_d=t.stack([x[0]for x in tv.datasets.CIFAR10(root='data/cifar10',transform=transform, download=False)]).to(device)
noise=lambda x : x+sigma*t.randn_like(x)
def sample_p_d():
    #Random uniform draws from dataset
    p_d_i=t.LongTensor(m).random_(0,p_d.shape[0])
    return noise(p_d[p_d_i]).detach()

print("==> Sampling...")
sample_p_0=lambda:t.FloatTensor(m,n_ch,im_sz,im_sz).uniform_(-1,1).to(device)


def sample_q(K=K):
    #Langevin Dynamics update to sample from EBM
    x_k=t.autograd.Variable(sample_p_0(),requires_grad=True)
    for k in range(K):
        #compute gradient term
        f_prime=t.autograd.grad(f(x_k).sum(),[x_k],retain_graph=True)[0]
        #add Gaussian noise (Langevin)
        x_k.data+=f_prime+1e-2*t.randn_like(x_k) 
    return x_k.detach()


#New MCMC method
def sample_q_new(K=K):
    x_k=t.autograd.Variable(sample_p_0(),requires_grad=True)
    for k in range(K):
        #compute MAP of target distribution (MAP of log p actually so map of neural net f)
        f_prime=t.autograd.grad(f(x_k).sum(),[x_k],retain_graph=True)[0]

        # f_map = 

        #MH proposal centered around MAP
        x_k.data+=f_prime+1e-2*t.randn_like(x_k) 

        # cov_map = 
        # x_k.data+=f_map+cov_map*t.randn_like(x_k) 

    return x_k.detach()

sqrt= lambda x : int(t.sqrt(t.Tensor([x])))
plot= lambda p,x : tv.utils.save_image(t.clamp(x,-1.,1.),p,normalize=True,nrow=sqrt(m))

print("==> Chosing Optimizer...")
optim=t.optim.Adam(f.parameters(),lr=1e-4,betas=[.9,.999])

print("==> Start Training...")
for i in range(n_i) :
    #sample from data distribution and EBM (Langevin)
    x_p_d,x_q=sample_p_d(),sample_q()
    # x_p_d,x_q=sample_p_d(),sample_q_new()
    
    #log likelihood (to maximize) and before taking gradient
    #mean() to average over all samples
    L=f(x_p_d).mean()-f(x_q).mean() 
    optim.zero_grad()
    #-L to minimize (SGD step)
    (-L).backward()
    optim.step()
    if i%100 == 0:
        print("==> Checkpoint")
        print('{:>6d}f(x_p_d)={:>14.9f}f(x_q)={:>14.9f}'.format(i,f(x_p_d).mean(),f(x_q).mean()))
        plot('x_q_{:>06d}.png'.format(i),x_q)

print("==> Generation is Finished...")

