import numpy as np
import torch
import math

class Example0a:
    """
    Cause and effect of a target with heteroskedastic noise
    """

    def __init__(self, dim_inv, dim_spu, n_envs):
        self.scramble = torch.eye(dim_inv + dim_spu)
        self.dim_inv = dim_inv
        self.dim_spu = dim_spu
        self.dim = dim_inv + dim_spu
        self.task = "regression"
        self.envs = {'E0': 0.1}

        if n_envs >= 2:
            self.envs['E1'] = 1.5
        if n_envs >= 3:
            self.envs["E2"] = 2
        if n_envs > 3:
            for env in range(3, n_envs):
                var = 10 ** torch.zeros(1).uniform_(-2, 1).item()
                self.envs["E" + str(env)] = var
        print("Environments variables:", self.envs)

        self.wxy = torch.randn(self.dim_inv, self.dim_inv) / self.dim_inv
        self.wyz = torch.randn(self.dim_inv, self.dim_spu) / self.dim_spu

    def sample(self, n=1000, env="E0", split="train"):
        sdv = self.envs[env]
        x = torch.randn(n, self.dim_inv) * sdv
        y = x @ self.wxy + torch.randn(n, self.dim_inv)*0.1 # got rid of *sdv otherwwise we don't have invariance
        z = y @ self.wyz + torch.randn(n, self.dim_spu)

        if split == "test":
            z = z[torch.randperm(len(z))]

        inputs = torch.cat((x, z), -1) @ self.scramble
        outputs = y.sum(1, keepdim=True)

        return inputs, outputs



class Example1:
    """
    Cause and effect of a target with heteroskedastic noise
    """

    def __init__(self, dim_inv, dim_spu, n_envs):
        self.scramble = torch.eye(dim_inv + dim_spu)
        self.dim_inv = dim_inv
        self.dim_spu = dim_spu
        self.dim = dim_inv + dim_spu

        self.task = "regression"
        self.envs = {'E0': 0.1}

        if n_envs >= 2:
            self.envs['E1'] = 1.5
        if n_envs >= 3:
            self.envs["E2"] = 2
        if n_envs > 3:
            for env in range(3, n_envs):
                var = 10 ** torch.zeros(1).uniform_(-2, 1).item()
                self.envs["E" + str(env)] = var
        print("Environments variables:", self.envs)

        self.wxy = torch.randn(self.dim_inv, self.dim_inv) / self.dim_inv
        self.wyz = torch.randn(self.dim_inv, self.dim_spu) / self.dim_spu

    def sample(self, n=1000, env="E0", split="train"):
        sdv = self.envs[env]
        x = torch.randn(n, self.dim_inv) * sdv
        y = x @ self.wxy + torch.randn(n, self.dim_inv) # got rid of *sdv otherwwise we don't have invariance
        z = y @ self.wyz + torch.randn(n, self.dim_spu)

        if split == "test":
            z = z[torch.randperm(len(z))]

        inputs = torch.cat((x, z), -1) @ self.scramble
        outputs = y.sum(1, keepdim=True)

        return inputs, outputs


class Example2:
    """
    Cows and camels
    """

    def __init__(self, dim_inv, dim_spu, n_envs):
        self.scramble = torch.eye(dim_inv + dim_spu)
        self.dim_inv = dim_inv
        self.dim_spu = dim_spu
        self.dim = dim_inv + dim_spu

        self.task = "classification"
        self.envs = { 'E0': {"p": 0.95, "s": 0.3} }

        if n_envs >= 2:
            self.envs['E1'] = {"p": 0.97, "s": 0.5}
        if n_envs >= 3:
            self.envs["E2"] = {"p": 0.99, "s": 0.7}
        if n_envs > 3:
            for env in range(3, n_envs):
                self.envs["E" + str(env)] = {
                    "p": torch.zeros(1).uniform_(0.9, 1).item(),
                    "s": torch.zeros(1).uniform_(0.3, 0.7).item()
                }
        print("Environments variables:", self.envs)

        # foreground is 100x noisier than background
        self.snr_fg = 0.01
        self.snr_bg = 1

        # foreground (fg) denotes animal (cow / camel)
        cow = torch.ones(1, self.dim_inv)
        self.avg_fg = torch.cat((cow, cow, -cow, -cow))

        # background (bg) denotes context (grass / sand)
        grass = torch.ones(1, self.dim_spu)
        self.avg_bg = torch.cat((grass, -grass, -grass, grass))

    def sample(self, n=1000, env="E0", split="train"):
        p = self.envs[env]["p"]
        s = self.envs[env]["s"]
        w = torch.Tensor([p, 1 - p] * 2) * torch.Tensor([s] * 2 + [1 - s] * 2)
        i = torch.multinomial(w, n, True)
        x = torch.cat((
            (torch.randn(n, self.dim_inv) /
                math.sqrt(10) + self.avg_fg[i]) * self.snr_fg,
            (torch.randn(n, self.dim_spu) /
                math.sqrt(10) + self.avg_bg[i]) * self.snr_bg), -1)

        if split == "test":
            x[:, self.dim_spu:] = x[torch.randperm(len(x)), self.dim_spu:]

        inputs = x @ self.scramble
        outputs = x[:, :self.dim_inv].sum(1, keepdim=True).gt(0).float()

        return inputs, outputs


class Example3:
    """
    Small invariant margin versus large spurious margin
    """

    def __init__(self, dim_inv, dim_spu, n_envs):
        self.scramble = torch.eye(dim_inv + dim_spu)
        self.dim_inv = dim_inv
        self.dim_spu = dim_spu
        self.dim = dim_inv + dim_spu

        self.task = "classification"
        self.envs = {}

        for env in range(n_envs):
            self.envs["E" + str(env)] = torch.randn(1, dim_spu)

        print("Environments variables:", self.envs)

    def sample(self, n=1000, env="E0", split="train"):
        m = n // 2
        sep = .1

        invariant_0 = torch.randn(m, self.dim_inv) * .1 + \
            torch.Tensor([[sep] * self.dim_inv])
        invariant_1 = torch.randn(m, self.dim_inv) * .1 - \
            torch.Tensor([[sep] * self.dim_inv])

        shortcuts_0 = torch.randn(m, self.dim_spu) * .1 + self.envs[env]
        shortcuts_1 = torch.randn(m, self.dim_spu) * .1 - self.envs[env]

        x = torch.cat((torch.cat((invariant_0, shortcuts_0), -1),
                       torch.cat((invariant_1, shortcuts_1), -1)))

        if split == "test":
            x[:, self.dim_inv:] = x[torch.randperm(len(x)), self.dim_inv:]

        inputs = x @ self.scramble
        outputs = torch.cat((torch.zeros(m, 1), torch.ones(m, 1)))

        return inputs, outputs

    
class Example4:
    """
    Cause and effect of a target: with mixed-type input variables
    Based on Example1, however the inv vars have fixed var, and the spu vars
    variance varies between environment (which is opposite to Ex1.)
    """
    def __init__(self, dim_inv, dim_spu, n_envs, p_inv=0.5):
        
        assert len(dim_inv) == 2
        assert len(dim_spu) == 2
        
        self.dim_inv_c, self.dim_inv_b = dim_inv
        self.dim_spu_c, self.dim_spu_b = dim_spu
        self.dim = self.dim_inv_c + self.dim_inv_b + self.dim_spu_c + self.dim_spu_b

        self.scramble = torch.eye(self.dim)
        self.p_inv = p_inv
        self.task = "regression"
        self.envs = {'E0': 0.3}
        if n_envs >= 2:
            self.envs["E1"] = 0.7
        if n_envs >= 3:
            self.envs["E2"] = 0.5
        if n_envs > 3:
            for env in range(3, n_envs):
                var = 0.25*torch.zeros(1).uniform_(0, 1).item()
                self.envs["E" + str(env)] = var

        print("Environments variables:", self.envs)

        self.dim_y = self.dim_inv_c+self.dim_inv_b
        self.wxy = torch.randn(self.dim_inv_c+self.dim_inv_b, self.dim_y) / self.dim_y
        self.wyz_c = torch.randn(self.dim_y, self.dim_spu_c) / self.dim_spu_c
        self.wyz_b = torch.randn(self.dim_y, self.dim_spu_b) / self.dim_spu_b

    def sample(self, n=1000, env="E0", split="train"):
        sdv = self.envs[env]
        x_c = torch.randn(n, self.dim_inv_c) * np.sqrt((self.p_inv)*(1-self.p_inv))
        x_b = torch.bernoulli(self.p_inv*torch.ones(n, self.dim_inv_b))
        x = torch.cat((x_c,x_b), dim=1)
        y = x @ self.wxy + torch.randn(n, self.dim_y) * np.sqrt((self.p_inv)*(1-self.p_inv))
        z_c = y @ self.wyz_c + torch.randn(n, self.dim_spu_c) * 10**(sdv*3-2)
        z_b = torch.remainder(
                    torch.ceil(y @ self.wyz_b + torch.bernoulli(sdv*torch.ones(n, self.dim_spu_b))
                              )
                        , 2)

        z = torch.cat((z_c, z_b), -1)
        if split == "test":
            z = z[torch.randperm(len(z))]

        inputs = torch.cat((x, z), -1) @ self.scramble
        outputs = y.sum(1, keepdim=True)
        return inputs, outputs


class Example5:
    """
    Cows and camels and fish (multi-class)
    """

    def __init__(self, dim_inv, dim_spu, n_envs):
        self.scramble = torch.eye(dim_inv + dim_spu)
        self.dim_inv = dim_inv
        self.dim_spu = dim_spu
        self.dim = dim_inv + dim_spu

        self.task = "multi-class classification"
        # p: p_1 needs to be high, to force values to the diagonal (ie, animals on their own backgrounds)
        # s: probability of each background 
        
        self.envs = {'E0': {"p": [0.8, 0.1, 0.1], "s": [0.3, 0.4, 0.3]}}
        
        if n_envs >= 2:
            self.envs['E1'] = {"p": [0.9, 0.05, 0.05], "s": [0.4, 0.3, 0.3]}
        if n_envs >= 3:
            self.envs["E2"] = {"p": [0.97, 0.01, 0.02], "s": [0.3, 0.3, 0.4]}
        if n_envs > 3:
            for env in range(3, n_envs):
                # bowl shaped simplex: i.e. non-uniform cat variables
                m1 = torch.distributions.Dirichlet(torch.Tensor([.9,.9,.9])) # peaked (somewhere - remember to sort samples)
                m2 = torch.distributions.Dirichlet(torch.Tensor([10,10,10])) # flat
#                 m = torch.distributions.Dirichlet(torch.Tensor([.5,.5,.5]))
                self.envs["E" + str(env)] = {
                "p": sorted(m1.sample().tolist(), reverse=True),
                "s": m2.sample().tolist()
                }
        print("Environments variables:", self.envs)

        # foreground is 100x noisier than background
        self.snr_fg = 1e-2  # nu_animal 
        self.snr_bg = 1     # nu_background

        # foreground (fg) denotes animal (cow / camel / fish)
        cow = torch.ones(1, self.dim_inv)
        camel = -torch.ones(1, self.dim_inv)
        fish = torch.zeros(1, self.dim_inv)
        
        self.avg_fg = torch.cat((cow, cow, cow, camel, camel, camel, fish, fish, fish))

        # background (bg) denotes context (grass / sand / water)
        grass = torch.ones(1, self.dim_spu)
        sand = -torch.ones(1, self.dim_spu)
        water = torch.zeros(1, self.dim_spu)
        
        self.avg_bg = torch.cat((grass, sand, water, sand, water, grass, water, grass, sand))

    def sample(self, n=1000, env="E0", split="train"):
        p = self.envs[env]["p"]
        s = self.envs[env]["s"]
        # w relates to the probabilities of: self.avg_fg * self.avg_bg. ie. creating high probable cows on grass, camels on sand etc.
        w = torch.Tensor(p * 3) * torch.Tensor([s[0]] * 3 + [s[2]] * 3 + [s[2]] * 3)
        i = torch.multinomial(w, n, True)
        x = torch.cat((
            (torch.randn(n, self.dim_inv) /
                math.sqrt(10) + self.avg_fg[i]) * self.snr_fg,
            (torch.randn(n, self.dim_spu) /
                math.sqrt(10) + self.avg_bg[i]) * self.snr_bg), -1)

        if split == "test":
            x[:, self.dim_spu:] = x[torch.randperm(len(x)), self.dim_spu:]

        inputs = x @ self.scramble
        outputs = x[:, :self.dim_inv].sum(1, keepdim=True).gt(0.005).float() + 2*x[:, :self.dim_inv].sum(1, keepdim=True).lt(-0.005).float()

        return inputs, outputs

class Example6:
    """
    Cause and effect of a target with heteroskedastic noise
    """

    def __init__(self, dim_inv, dim_spu, n_envs):
        self.scramble = torch.eye(dim_inv + dim_spu)
        self.dim_inv = dim_inv
        self.dim_spu = dim_spu
        self.dim = dim_inv + dim_spu

        self.task = "regression"
        self.envs = {'E0': 0.1}

        if n_envs >= 2:
            self.envs['E1'] = 1.5
        if n_envs >= 3:
            self.envs["E2"] = 2
        if n_envs > 3:
            for env in range(3, n_envs):
                var = 10 ** torch.zeros(1).uniform_(-2, 1).item()
                self.envs["E" + str(env)] = var
        print("Environments variables:", self.envs)

        self.wxy = torch.randn(self.dim_inv, self.dim_inv) / self.dim_inv
        self.wyz = torch.randn(self.dim_inv, self.dim_spu) / self.dim_spu

    def sample(self, n=1000, env="E0", split="train"):
        sdv = self.envs[env]
        x = torch.randn(n, self.dim_inv) * sdv
        y = x @ self.wxy + torch.randn(n, self.dim_inv) * sdv # got rid of *sdv otherwwise we don't have invariance
        z = y @ self.wyz + torch.randn(n, self.dim_spu)

        if split == "test":
            z = z[torch.randperm(len(z))]

        inputs = torch.cat((x, z), -1) @ self.scramble
        outputs = y.sum(1, keepdim=True)

        return inputs, outputs
    

class Example1s(Example1):
    def __init__(self, dim_inv, dim_spu, n_envs):
        super().__init__(dim_inv, dim_spu, n_envs)

        self.scramble, _ = torch.qr(torch.randn(self.dim, self.dim))


class Example2s(Example2):
    def __init__(self, dim_inv, dim_spu, n_envs):
        super().__init__(dim_inv, dim_spu, n_envs)

        self.scramble, _ = torch.qr(torch.randn(self.dim, self.dim))


class Example3s(Example3):
    def __init__(self, dim_inv, dim_spu, n_envs):
        super().__init__(dim_inv, dim_spu, n_envs)

        self.scramble, _ = torch.qr(torch.randn(self.dim, self.dim))

class Example4s(Example4):
    def __init__(self, dim_inv, dim_spu, n_envs):
        super().__init__(dim_inv, dim_spu, n_envs)

        self.scramble, _ = torch.qr(torch.randn(self.dim, self.dim))
        
DATASETS = {
    "Example0a": Example0a,
    "Example0b": Example2,
    "Example1": Example1,
    "Example2": Example2,
    "Example3": Example3,
    "Example4": Example4,  # mixed-type input vars 
    "Example5": Example5, # multi-class version of Ex2
    "Example6": Example6, # regression with hidden variable
    "Example1s": Example1s,
    "Example2s": Example2s,
    "Example3s": Example3s,
}
