import torch
import torch.nn as nn
import random



class Similarity(nn.Module):
    """
    Dot product or cosine similarity
    """

    def __init__(self, temp):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x, y):
        return self.cos(x, y) / self.temp
    
class ContrastiveLoss(nn.Module):
    def __init__(self, temp=0.5):
        super().__init__()
        self.loss = nn.CrossEntropyLoss()
        self.sim_func = Similarity(temp=temp)

    def forward(self, emb1, emb2):
        batch_sim = self.sim_func(emb1.unsqueeze(1), emb2.unsqueeze(0))
        labels = torch.arange(batch_sim.size(0)).long().to('cuda')
        return self.loss(batch_sim, labels)


class CLUBSample(nn.Module):  # Sampled version of the CLUB estimator
    def __init__(self, x_dim, y_dim, hidden_size):
        super(CLUBSample, self).__init__()
        self.p_mu = nn.Sequential(
            nn.Linear(x_dim, hidden_size//2),
            nn.ReLU(),
            nn.Linear(hidden_size//2, y_dim)
        )

        self.p_logvar = nn.Sequential(
            nn.Linear(x_dim, hidden_size//2),
            nn.ReLU(),
            nn.Linear(hidden_size//2, y_dim),
            nn.Tanh()
        )

    def get_mu_logvar(self, x_samples):
        mu = self.p_mu(x_samples)
        logvar = self.p_logvar(x_samples)
        return mu, logvar
     
        
    def loglikeli(self, x_samples, y_samples):
        mu, logvar = self.get_mu_logvar(x_samples)
        return (-(mu - y_samples)**2 /2./logvar.exp()).sum(dim=1).mean(dim=0)
    

    def forward(self, x_samples, y_samples):
        mu, logvar = self.get_mu_logvar(x_samples)
        
        sample_size = x_samples.shape[0]
        #random_index = torch.randint(sample_size, (sample_size,)).long()
        random_index = torch.randperm(sample_size).long()
        
        positive = - (mu - y_samples)**2 / logvar.exp()
        negative = - (mu - y_samples[random_index])**2 / logvar.exp()
        upper_bound = (positive.sum(dim = -1) - negative.sum(dim = -1)).mean()
        return upper_bound / 2.0

    def learning_loss(self, x_samples, y_samples):
        return - self.loglikeli(x_samples, y_samples)


class MIEstimator(nn.Module):
    def __init__(self, args):
        super(MIEstimator, self).__init__()
        self.str_estimator = CLUBSample(args.dim, args.dim, args.dim)
        self.img_estimator = CLUBSample(args.img_dim, args.img_dim, args.img_dim)
        self.c_img_estimator = CLUBSample(args.img_dim, args.img_dim, args.img_dim)
        self.d_img_estimator = CLUBSample(args.diffusion_image_dim, args.diffusion_image_dim, args.diffusion_image_dim)
        self.txt_estimator = CLUBSample(args.txt_dim, args.txt_dim, args.txt_dim)
        self.c_txt_estimator = CLUBSample(args.txt_dim, args.txt_dim, args.txt_dim)
        self.num = args.n_exp
    

    def forward(self, embeddings, extra_embeddings=None):
        strs, imgs, txts = embeddings
        bzs, n_exp, _ = imgs.size()
        assert n_exp == self.num
        idx1, idx2 = random.sample(range(n_exp), k=2)
        
        # 基础三元组采样
        str1, str2 = strs[:, idx1, :], strs[:, idx2, :]
        img1, img2 = imgs[:, idx1, :], imgs[:, idx2, :]
        txt1, txt2 = txts[:, idx1, :], txts[:, idx2, :]
        
        mi_loss = 0.0
        count = 0  # 记录使用的估计器数量
        
        # 始终使用的基础估计器
        mi_loss += self.str_estimator(str1, str2)
        mi_loss += self.img_estimator(img1, img2)
        mi_loss += self.txt_estimator(txt1, txt2)
        count += 3
        
        # 处理额外估计器（可独立控制）
        if extra_embeddings is not None:
            c_imgs, c_txts, d_imgs = extra_embeddings
            if c_imgs is not None:
                c_img1, c_img2 = c_imgs[:, idx1, :], c_imgs[:, idx2, :]
                mi_loss += self.c_img_estimator(c_img1, c_img2)
                count += 1
            if c_txts is not None:
                c_txt1, c_txt2 = c_txts[:, idx1, :], c_txts[:, idx2, :]
                mi_loss += self.c_txt_estimator(c_txt1, c_txt2)
                count += 1
            if d_imgs is not None:
                d_imgs1, d_imgs2 = d_imgs[:, idx1, :], d_imgs[:, idx2, :]
                mi_loss += self.d_img_estimator(d_imgs1, d_imgs2)
                count += 1
        
        mi_loss = mi_loss / (2.0 * count)
        return mi_loss

    def train_estimator(self, embeddings, extra_embeddings=None):
        strs, imgs, txts = embeddings
        bzs, n_exp, _ = imgs.size()
        assert n_exp == self.num
        idx1, idx2 = random.sample(range(n_exp), k=2)
        
        str1, str2 = strs[:, idx1, :], strs[:, idx2, :]
        img1, img2 = imgs[:, idx1, :], imgs[:, idx2, :]
        txt1, txt2 = txts[:, idx1, :], txts[:, idx2, :]
        
        est_loss = 0.0
        count = 0
        
        # 基础估计器
        est_loss += self.str_estimator.learning_loss(str1, str2)
        est_loss += self.img_estimator.learning_loss(img1, img2)
        est_loss += self.txt_estimator.learning_loss(txt1, txt2)
        count += 3
        
        if extra_embeddings is not None:
            c_imgs, c_txts, d_imgs = extra_embeddings
            if c_imgs is not None:
                c_img1, c_img2 = c_imgs[:, idx1, :], c_imgs[:, idx2, :]
                est_loss += self.c_img_estimator.learning_loss(c_img1, c_img2)
                count += 1
            if c_txts is not None:
                c_txt1, c_txt2 = c_txts[:, idx1, :], c_txts[:, idx2, :]
                est_loss += self.c_txt_estimator.learning_loss(c_txt1, c_txt2)
                count += 1
            if d_imgs is not None:
                d_imgs1, d_imgs2 = d_imgs[:, idx1, :], d_imgs[:, idx2, :]
                est_loss += self.d_img_estimator.learning_loss(d_imgs1, d_imgs2)
                count += 1
        
        est_loss = est_loss / (2.0 * count)
        return est_loss

