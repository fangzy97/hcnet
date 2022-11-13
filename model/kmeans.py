import numpy as np
import torch


class KmeansClustering:
    def __init__(self, num_cnt, iters, init='kmeans++'):
        self.num_cnt = num_cnt
        self.iters = iters
        self.init_mode = init

    def init_func(self, x):
        N, D = x.shape
        if self.init_mode == 'random':
            random_inds = np.random.choice(N, self.num_cnt)
            init_stat = x[random_inds]  # num_cnt, D

        elif self.init_mode == 'kmeans++':
            random_start = np.random.choice(N, 1)
            init_stat = x[[random_start]]  # 1*D
            x1 = x.unsqueeze(1)
            x_lazy = x1 # N*1*D
            for c_id in range(self.num_cnt - 1):
                init_lazy = init_stat.unsqueeze(0)  # 1*M*D
                dist = ((x_lazy - init_lazy) ** 2).sum(-1)  # N*M
                select = dist.min(1)[0].view(-1).argmax(dim=0)
                init_stat = torch.cat((init_stat, x1[select]), dim=0)
        else:
            raise NotImplementedError

        return init_stat

    def cluster(self, x, center=None):
        with torch.no_grad():
            if center is None:
                center = self.init_func(x)  # M*D

            x_lazy = x.unsqueeze(1)  # (N,1,D)
            cl = None
            for iter in range(self.iters):
                c_lazy = center.unsqueeze(0)  # 1*M*D

                dist = ((x_lazy - c_lazy) ** 2).sum(-1)  # N*M

                cl = dist.argmin(dim=1).view(-1)

                if iter < self.iters - 1:
                    for cnt_id in range(self.num_cnt):
                        selected = torch.nonzero(cl == cnt_id, as_tuple=False).squeeze(-1)
                        if selected.shape[0] != 0:
                            selected = torch.index_select(x, 0, selected)
                            center[cnt_id] = selected.mean(dim=0)

            center_new = torch.zeros_like(center).to(torch.device('cuda'))
            for cnt_id in range(self.num_cnt):
                selected = torch.nonzero(cl == cnt_id, as_tuple=False).squeeze(-1)
                if selected.shape[0] != 0:
                    selected = torch.index_select(x, 0, selected)
                    center_new[cnt_id] = selected.mean(dim=0)

        return center_new
