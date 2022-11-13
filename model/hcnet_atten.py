import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import BatchNorm2d as BatchNorm

from .backbone import resnet as models
from .kmeans import KmeansClustering
from .modules.ASPP import ASPP
from .modules.decoder import build_decoder


# Masked Average Pooling
def Weighted_GAP(supp_feat, mask):
    supp_feat = supp_feat * mask
    feat_h, feat_w = supp_feat.shape[-2:]
    area = F.avg_pool2d(mask, (feat_h, feat_w)) * feat_h * feat_w + 0.0005
    supp_feat = F.avg_pool2d(input=supp_feat, kernel_size=supp_feat.shape[-2:]) * feat_h * feat_w / area
    return supp_feat


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()

        layers = args.layers
        classes = args.classes
        pretrained = True
        assert layers in [50, 101, 152]
        assert classes > 1

        self.zoom_factor = args.zoom_factor
        self.criterion = nn.CrossEntropyLoss(ignore_index=255)
        self.aux_criterion = nn.BCEWithLogitsLoss()
        self.shot = args.shot
        self.train_iter = args.train_iter
        self.eval_iter = args.eval_iter
        self.pyramid = args.pyramid
        self.freeze = args.freeze_bn

        models.BatchNorm = BatchNorm

        print('INFO: Using ResNet {}'.format(layers))
        if layers == 50:
            resnet = models.resnet50(pretrained=pretrained)
        elif layers == 101:
            resnet = models.resnet101(pretrained=pretrained)
        else:
            resnet = models.resnet152(pretrained=pretrained)
        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu1, resnet.conv2, resnet.bn2, resnet.relu2,
                                    resnet.conv3, resnet.bn3, resnet.relu3, resnet.maxpool)
        self.layer1, self.layer2, self.layer3, self.layer4 = resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4

        for n, m in self.layer3.named_modules():
            if 'conv2' in n:
                m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
            elif 'downsample.0' in n:
                m.stride = (1, 1)
        for n, m in self.layer4.named_modules():
            if 'conv2' in n:
                m.dilation, m.padding, m.stride = (4, 4), (4, 4), (1, 1)
            elif 'downsample.0' in n:
                m.stride = (1, 1)

        reduce_dim = 256
        fea_dim = 1024 + 512
        self.debug_list = []

        for param in self.parameters():
            param.requires_grad = False

        self.w_fg = nn.Parameter(torch.ones(reduce_dim, reduce_dim))
        self.w_bg = nn.Parameter(torch.ones(reduce_dim, reduce_dim), requires_grad=False)
        self.t = nn.Parameter(torch.tensor([0.07]))

        self.cls = nn.Sequential(
            nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(16, reduce_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduce_dim, classes, kernel_size=1)
        )

        self.down_conv = nn.Sequential(
            nn.Conv2d(fea_dim, reduce_dim, kernel_size=1, padding=0, bias=False),
            nn.GroupNorm(16, reduce_dim)
        )

        self.combine1 = nn.Sequential(
            nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(16, reduce_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False)
        )

        self.combine4 = nn.Sequential(
            nn.Conv2d(reduce_dim * 4, reduce_dim, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(16, reduce_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False)
        )

        self.combine9 = nn.Sequential(
            nn.Conv2d(reduce_dim * 9, reduce_dim, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(16, reduce_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False)
        )

        self.combine16 = nn.Sequential(
            nn.Conv2d(reduce_dim * 16, reduce_dim, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(16, reduce_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False)
        )

        self.combine_bg = nn.Sequential(
            nn.Conv2d(reduce_dim * 30, reduce_dim, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(16, reduce_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False)
        )

        self.ASPP = ASPP(out_channels=reduce_dim)
        self.corr_conv = nn.Sequential(
            nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False)
        )

        self.skip1 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(16, reduce_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(16, reduce_dim)
        )
        self.skip2 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(16, reduce_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(16, reduce_dim)
        )
        self.skip3 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(16, reduce_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(16, reduce_dim)
        )
        self.decoder = build_decoder(256)
        self.cls_aux = nn.Sequential(nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
                                     nn.GroupNorm(16, reduce_dim),
                                     nn.ReLU(inplace=True),
                                     nn.Conv2d(reduce_dim, classes, kernel_size=1))

    def freeze_bn(self):
        for n, m in self.named_modules():
            if 'layer' in n and isinstance(m, nn.BatchNorm2d):
                m.eval()

    def forward(self, x, s_x, s_y, y, cat_idx=None):
        if self.freeze:
            self.freeze_bn()

        x_size = x.size()
        assert (x_size[2] - 1) % 8 == 0 and (x_size[3] - 1) % 8 == 0
        h = int((x_size[2] - 1) / 8 * self.zoom_factor + 1)
        w = int((x_size[3] - 1) / 8 * self.zoom_factor + 1)

        # Query Feature
        with torch.no_grad():
            query_feat_0 = self.layer0(x)
            query_feat_1 = self.layer1(query_feat_0)
            query_feat_2 = self.layer2(query_feat_1)
            query_feat_3 = self.layer3(query_feat_2)

        query_feat = torch.cat([query_feat_3, query_feat_2], 1)
        query_feat = self.down_conv(query_feat)

        hide_size = query_feat_1.shape[-2:]

        # Support Feature
        supp_feat_list = []
        mask_list = []
        bg_mask_list = []
        for i in range(self.shot):
            mask = (s_y[:, i, :, :] == 1).float().unsqueeze(1)
            bg_mask = (s_y[:, i, :, :] == 0).float().unsqueeze(1)
            with torch.no_grad():
                supp_feat_0 = self.layer0(s_x[:, i, :, :, :])
                supp_feat_1 = self.layer1(supp_feat_0)
                supp_feat_2 = self.layer2(supp_feat_1)
                supp_feat_3 = self.layer3(supp_feat_2)

            supp_feat = torch.cat([supp_feat_3, supp_feat_2], 1)
            supp_feat = self.down_conv(supp_feat)
            supp_feat = F.interpolate(supp_feat, hide_size, mode='bilinear', align_corners=True)
            mask = F.interpolate(mask, hide_size, mode='bilinear', align_corners=True)
            bg_mask = F.interpolate(bg_mask, hide_size, mode='bilinear', align_corners=True)
            supp_feat_list.append(supp_feat)  # shot x [bs x 256 x h x w]
            mask_list.append(mask)
            bg_mask_list.append(bg_mask)

        # ========  masked average pooling ======== #
        m_vec_shot = []
        for supp_feat, mask in zip(supp_feat_list, mask_list):
            m_vec = Weighted_GAP(supp_feat, mask)
            m_vec_shot.append(m_vec)
        m_vec_shot = sum(m_vec_shot) / len(m_vec_shot)
        sim_matrix = torch.cosine_similarity(query_feat, m_vec_shot)
        self.debug_maxmin("sim_martix",sim_matrix)
        sim_map = query_feat * sim_matrix.unsqueeze(1)
        query_feat = self.combine1(sim_map)

        # ======== k-means cluster ======== #
        cluster_4 = []
        cluster_9 = []
        cluster_16 = []
        cluster_bg = []
        for supp_feat, mask, bg_mask in zip(supp_feat_list, mask_list, bg_mask_list):
            cluster_4 += [self.cluster_pipeline(supp_feat, mask, 4)]
            cluster_9 += [self.cluster_pipeline(supp_feat, mask, 9)]
            cluster_16 += [self.cluster_pipeline(supp_feat, mask, 16)]
            cluster_bg += [self.cluster_pipeline(supp_feat, bg_mask, 1 + 4 + 9 + 16)]
        cluster_4 = sum(cluster_4) / len(cluster_4)  # b x k x c
        cluster_9 = sum(cluster_9) / len(cluster_9)  # b x k x c
        cluster_16 = sum(cluster_16) / len(cluster_16)
        cluster_bg = sum(cluster_bg) / len(cluster_bg)
        # cluster_4 = torch.cat(cluster_4, dim=1)  # b x (shot x 4) x c
        # cluster_9 = torch.cat(cluster_9, dim=1)  # b x (shot x 9) x c

        query_feat, vis_map4 = self.metric_pipeline(cluster_4, query_feat, self.combine4)
        query_feat, vis_map9 = self.metric_pipeline(cluster_9, query_feat, self.combine9)
        query_feat, vis_map16 = self.metric_pipeline(cluster_16, query_feat, self.combine16)
        query_feat, _ = self.metric_pipeline(-cluster_bg, query_feat, self.combine_bg)

        out = self.aspp_pipeline(query_feat, query_feat_1)

        if self.zoom_factor != 1:
            out = F.interpolate(out, size=(h, w), mode='bilinear', align_corners=True)

        if self.training:
            main_loss = self.criterion(out, y.long())

            # cluster是聚类得到的 m 个前景特征向量
            cluster = torch.cat((m_vec_shot.squeeze(-1).squeeze(-1).unsqueeze(1), cluster_4, cluster_9, cluster_16), dim=1)
            # w_fg是一个可学习的权重矩阵W，下面的w_bg同。w_fg通过梯度下降更新，而w_bg通过动量更新的方式更新，具体可见moco和moco v2
            cluster = torch.einsum('bic,cd->bid', cluster, self.w_fg)
            cluster = F.normalize(cluster, p=2, dim=2)
            # 这里计算得到各个前景特征向量之间的相似度矩阵
            logits_pos = torch.einsum('bic,bjc->bij', cluster, cluster.detach()).unsqueeze(-1)  # b x m x m
            with torch.no_grad():
                # 先更新一下w_bg，对应moco里更新encoder key的更新
                self.w_bg.data = 0.9 * self.w_bg.data + 0.1 * self.w_fg.data
                # cluster_bg是聚类得到的 n 个背景的特征向量
                cluster_bg = torch.einsum('bic,cd->bid', cluster_bg, self.w_bg)
                cluster_bg = F.normalize(cluster_bg, p=2, dim=2)
                # 这里计算前景特征向量和背景特征向量的相似度矩阵
                logits_neg = torch.einsum('bic,bjc->bij', cluster, cluster_bg.detach())  # b x m x n
                logits_neg = logits_neg.unsqueeze(2).expand(-1, -1, logits_pos.shape[2], -1)
            # 计算infoNCE。因为有m x m个正样本对，所以这里的计算矩阵是 m x m x (n + 1)
            logits = torch.cat((logits_pos, logits_neg), dim=-1) / self.t  # b x m x m x (n + 1)
            logits = -torch.log(torch.softmax(logits, dim=-1))
            aux_loss = torch.mean(logits[:, :, :, 0])

            return out.argmax(dim=1), main_loss, aux_loss
        else:
            return out

    def cluster_pipeline(self, s_feat, s_mask, cluster_num):
        b, c, h, w = s_feat.shape
        t_feat = s_feat.view(b, c, -1).transpose(1, 2)
        t_mask = s_mask.view(b, -1)

        cluster_out = []
        for i in range(b):
            fg_pixel = t_feat[i, t_mask[i] == 1]
            if fg_pixel.shape[0] == 0:
                fg_pixel = t_feat[i]
            cluster_out += [KmeansClustering(cluster_num, 20).cluster(fg_pixel)]
        out = torch.stack(cluster_out)  # b x k x c
        return out

    def metric_pipeline(self, s_vec, q_feat, combine_fn):
        out = []
        for i in range(s_vec.shape[1]):
            sim_matrix = torch.cosine_similarity(q_feat, s_vec[:, i, :, None, None])
            sim_matrix = q_feat * sim_matrix.unsqueeze(1)
            out.append(sim_matrix)
        out = torch.cat(out, dim=1)
        out = combine_fn(out)

        vis_map = []
        if not self.training:
            for i in range(s_vec.shape[1]):
                sim_matrix = torch.cosine_similarity(q_feat, s_vec[:, i, :, None, None])
                vis_map.append(sim_matrix)
            vis_map = sum(vis_map)  # b x h x w
            # vis_map = (vis_map - vis_map.mean(dim=(1, 2), keepdim=True)) / (vis_map.std(dim=(1, 2), keepdim=True) + 1e-5)

        return out, vis_map

    def aspp_pipeline(self, feat, low_level_feat):
        final_feat = self.corr_conv(feat)
        final_feat = final_feat + self.skip1(final_feat)
        final_feat = final_feat + self.skip2(final_feat)
        final_feat = final_feat + self.skip3(final_feat)
        final_feat = self.ASPP(final_feat)
        decoder_out = self.decoder(final_feat, low_level_feat)
        out = self.cls(decoder_out)
        return out

    def _optimizer(self, args):
        optimizer = torch.optim.SGD(filter(lambda x: x.requires_grad, self.parameters()),
                                    lr=args.base_lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)
        return optimizer

    def debug_maxmin(self,str,var):
        if str not in self.debug_list:
            print(str+": max {} min {} mean {}".format(var.max(),var.min(),var.mean()))
            self.debug_list.append(str)