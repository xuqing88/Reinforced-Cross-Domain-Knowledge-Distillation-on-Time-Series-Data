import torch
import torch.nn as nn
import numpy as np
import sys

from models.models import classifier, ReverseLayerF, Discriminator, RandomLayer, Discriminator_CDAN, \
    codats_classifier, AdvSKM_Disc, Discriminator_fea, Adapter,Discriminator_t,classifier_T,Discriminator_s
from models.loss import MMD_loss, CORAL, ConditionalEntropyLoss, VAT, LMMD_loss, HoMM_loss
from utils import EMA, jitter

from torch.autograd import Variable
from replay_memory import ReplayMemory
import torch.nn.functional as F

from sklearn.metrics.pairwise import cosine_distances,euclidean_distances
from scipy.special import kl_div


def get_algorithm_class(algorithm_name):
    """Return the algorithm class with the given name."""
    if algorithm_name not in globals():
        raise NotImplementedError("Algorithm not found: {}".format(algorithm_name))
    return globals()[algorithm_name]


class Algorithm(torch.nn.Module):
    """
    A subclass of Algorithm implements a domain adaptation algorithm.
    Subclasses should implement the update() method.
    """

    def __init__(self, configs):
        super(Algorithm, self).__init__()
        self.configs = configs
        self.cross_entropy = nn.CrossEntropyLoss()

    def update(self, *args, **kwargs):
        raise NotImplementedError


class Lower_Upper_bounds(Algorithm):
    """
    Lower bound: train on source and test on target.
    Upper bound: train on target and test on target.
    """

    def __init__(self, backbone_fe, configs, hparams, device):
        super(Lower_Upper_bounds, self).__init__(configs)

        self.feature_extractor = backbone_fe(configs)
        self.classifier = classifier_T(configs)
        self.network = nn.Sequential(self.feature_extractor, self.classifier)

        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )
        self.hparams = hparams

    def update(self, src_x, src_y):
        src_feat = self.feature_extractor(src_x)
        src_pred = self.classifier(src_feat)

        src_cls_loss = self.cross_entropy(src_pred, src_y)

        loss = src_cls_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'Src_cls_loss': src_cls_loss.item()}


class Deep_Coral(Algorithm):
    """
    Deep Coral: https://arxiv.org/abs/1607.01719
    """
    def __init__(self, backbone_fe, configs, hparams, device):
        super(Deep_Coral, self).__init__(configs)

        self.coral = CORAL()

        self.feature_extractor = backbone_fe(configs)
        self.classifier = classifier(configs)
        self.network = nn.Sequential(self.feature_extractor, self.classifier)

        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )
        self.hparams = hparams

    def update(self, src_x, src_y, trg_x):
        src_feat = self.feature_extractor(src_x)
        src_pred = self.classifier(src_feat)

        src_cls_loss = self.cross_entropy(src_pred, src_y)

        trg_feat = self.feature_extractor(trg_x)

        coral_loss = self.coral(src_feat, trg_feat)

        loss = self.hparams["coral_wt"] * coral_loss + \
               self.hparams["src_cls_loss_wt"] * src_cls_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'Total_loss': loss.item(), 'Coral_loss': coral_loss.item(), 'Src_cls_loss': src_cls_loss.item()}


class MMDA(Algorithm):
    """
    MMDA: https://arxiv.org/abs/1901.00282
    """

    def __init__(self, backbone_fe, configs, hparams, device):
        super(MMDA, self).__init__(configs)

        self.mmd = MMD_loss()
        self.coral = CORAL()
        self.cond_ent = ConditionalEntropyLoss()

        self.feature_extractor = backbone_fe(configs)
        self.classifier = classifier(configs)
        self.network = nn.Sequential(self.feature_extractor, self.classifier)

        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )
        self.hparams = hparams

    def update(self, src_x, src_y, trg_x):
        src_feat = self.feature_extractor(src_x)
        src_pred = self.classifier(src_feat)

        src_cls_loss = self.cross_entropy(src_pred, src_y)

        trg_feat = self.feature_extractor(trg_x)

        coral_loss = self.coral(src_feat, trg_feat)
        mmd_loss = self.mmd(src_feat, trg_feat)
        cond_ent_loss = self.cond_ent(trg_feat)

        loss = self.hparams["coral_wt"] * coral_loss + \
               self.hparams["mmd_wt"] * mmd_loss + \
               self.hparams["cond_ent_wt"] * cond_ent_loss + \
               self.hparams["src_cls_loss_wt"] * src_cls_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'Total_loss': loss.item(), 'Coral_loss': coral_loss.item(), 'MMD_loss': mmd_loss.item(),
                'cond_ent_wt': cond_ent_loss.item(), 'Src_cls_loss': src_cls_loss.item()}


class DANN(Algorithm):
    """
    DANN: https://arxiv.org/abs/1505.07818
    """

    def __init__(self, backbone_fe, configs, hparams, device):
        super(DANN, self).__init__(configs)

        self.feature_extractor = backbone_fe(configs)
        self.classifier = classifier(configs)
        self.network = nn.Sequential(self.feature_extractor, self.classifier)

        self.domain_classifier = Discriminator(configs)

        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"], betas=(0.5, 0.99)
        )
        self.optimizer_disc = torch.optim.Adam(
            self.domain_classifier.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"], betas=(0.5, 0.99)
        )
        self.hparams = hparams
        self.device = device

    def update(self, src_x, src_y, trg_x, step, epoch, len_dataloader):
        p = float(step + epoch * len_dataloader) / self.hparams["num_epochs"] + 1 / len_dataloader
        alpha = 2. / (1. + np.exp(-10 * p)) - 1

        # zero grad
        self.optimizer.zero_grad()
        self.optimizer_disc.zero_grad()

        domain_label_src = torch.ones(len(src_x)).to(self.device)
        domain_label_trg = torch.zeros(len(trg_x)).to(self.device)

        src_feat = self.feature_extractor(src_x)
        src_pred = self.classifier(src_feat)

        trg_feat = self.feature_extractor(trg_x)

        # Task classification  Loss
        src_cls_loss = self.cross_entropy(src_pred.squeeze(), src_y)

        # Domain classification loss
        # source
        src_feat_reversed = ReverseLayerF.apply(src_feat, alpha)
        src_domain_pred = self.domain_classifier(src_feat_reversed)
        src_domain_loss = self.cross_entropy(src_domain_pred, domain_label_src.long())

        # target
        trg_feat_reversed = ReverseLayerF.apply(trg_feat, alpha)
        trg_domain_pred = self.domain_classifier(trg_feat_reversed)
        trg_domain_loss = self.cross_entropy(trg_domain_pred, domain_label_trg.long())

        # Total domain loss
        domain_loss = src_domain_loss + trg_domain_loss

        loss = self.hparams["src_cls_loss_wt"] * src_cls_loss + \
               self.hparams["domain_loss_wt"] * domain_loss

        loss.backward()
        self.optimizer.step()
        self.optimizer_disc.step()

        return {'Total_loss': loss.item(), 'Domain_loss': domain_loss.item(), 'Src_cls_loss': src_cls_loss.item()}


class DANN_T(Algorithm):
    """
    DANN: https://arxiv.org/abs/1505.07818
    """

    def __init__(self, backbone_fe, configs, hparams, device):
        super(DANN_T, self).__init__(configs)

        self.feature_extractor = backbone_fe(configs)
        self.classifier = classifier_T(configs)
        self.network = nn.Sequential(self.feature_extractor, self.classifier)

        self.domain_classifier = Discriminator_t(configs)

        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"], betas=(0.5, 0.99)
        )
        self.optimizer_disc = torch.optim.Adam(
            self.domain_classifier.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"], betas=(0.5, 0.99)
        )
        self.hparams = hparams
        self.device = device

    def update(self, src_x, src_y, trg_x, step, epoch, len_dataloader):
        p = float(step + epoch * len_dataloader) / self.hparams["num_epochs"] + 1 / len_dataloader
        alpha = 2. / (1. + np.exp(-10 * p)) - 1

        # zero grad
        self.optimizer.zero_grad()
        self.optimizer_disc.zero_grad()

        domain_label_src = torch.ones(len(src_x)).to(self.device)
        domain_label_trg = torch.zeros(len(trg_x)).to(self.device)

        src_feat = self.feature_extractor(src_x)
        src_pred = self.classifier(src_feat)

        trg_feat = self.feature_extractor(trg_x)

        # Task classification  Loss
        src_cls_loss = self.cross_entropy(src_pred.squeeze(), src_y)

        # Domain classification loss
        # source
        src_feat_reversed = ReverseLayerF.apply(src_feat, alpha)
        src_domain_pred = self.domain_classifier(src_feat_reversed)
        src_domain_loss = self.cross_entropy(src_domain_pred, domain_label_src.long())

        # target
        trg_feat_reversed = ReverseLayerF.apply(trg_feat, alpha)
        trg_domain_pred = self.domain_classifier(trg_feat_reversed)
        trg_domain_loss = self.cross_entropy(trg_domain_pred, domain_label_trg.long())

        # Total domain loss
        domain_loss = src_domain_loss + trg_domain_loss

        loss = self.hparams["src_cls_loss_wt"] * src_cls_loss + \
               self.hparams["domain_loss_wt"] * domain_loss

        loss.backward()
        self.optimizer.step()
        self.optimizer_disc.step()

        return {'Total_loss': loss.item(), 'Domain_loss': domain_loss.item(), 'Src_cls_loss': src_cls_loss.item()}


class CDAN(Algorithm):
    """
    CDAN: https://arxiv.org/abs/1705.10667
    """

    def __init__(self, backbone_fe, configs, hparams, device):
        super(CDAN, self).__init__(configs)

        self.feature_extractor = backbone_fe(configs)
        self.classifier = classifier(configs)
        self.network = nn.Sequential(self.feature_extractor, self.classifier)

        self.domain_classifier = Discriminator_CDAN(configs)
        self.random_layer = RandomLayer([configs.features_len * configs.final_out_channels, configs.num_classes],
                                        configs.features_len * configs.final_out_channels)

        # optimizers
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )
        self.optimizer_disc = torch.optim.Adam(
            self.domain_classifier.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )
        # hparams
        self.hparams = hparams
        self.criterion_cond = ConditionalEntropyLoss().to(device)
        self.device = device

    def update(self, src_x, src_y, trg_x):
        # prepare true domain labels
        domain_label_src = torch.ones(len(src_x)).to(self.device)
        domain_label_trg = torch.zeros(len(trg_x)).to(self.device)
        domain_label_concat = torch.cat((domain_label_src, domain_label_trg), 0).long()

        # source features and predictions
        src_feat = self.feature_extractor(src_x)
        src_pred = self.classifier(src_feat)

        # target features and predictions
        trg_feat = self.feature_extractor(trg_x)
        trg_pred = self.classifier(trg_feat)

        # concatenate features and predictions
        feat_concat = torch.cat((src_feat, trg_feat), dim=0)
        pred_concat = torch.cat((src_pred, trg_pred), dim=0)

        # Domain classification loss
        feat_x_pred = torch.bmm(pred_concat.unsqueeze(2), feat_concat.unsqueeze(1)).detach()
        disc_prediction = self.domain_classifier(feat_x_pred.view(-1, pred_concat.size(1) * feat_concat.size(1)))
        disc_loss = self.cross_entropy(disc_prediction, domain_label_concat)

        # update Domain classification
        self.optimizer_disc.zero_grad()
        disc_loss.backward()
        self.optimizer_disc.step()

        # prepare fake domain labels for training the feature extractor
        domain_label_src = torch.zeros(len(src_x)).long().to(self.device)
        domain_label_trg = torch.ones(len(trg_x)).long().to(self.device)
        domain_label_concat = torch.cat((domain_label_src, domain_label_trg), 0)

        # Repeat predictions after updating discriminator
        feat_x_pred = torch.bmm(pred_concat.unsqueeze(2), feat_concat.unsqueeze(1))
        disc_prediction = self.domain_classifier(feat_x_pred.view(-1, pred_concat.size(1) * feat_concat.size(1)))
        # loss of domain discriminator according to fake labels

        domain_loss = self.cross_entropy(disc_prediction, domain_label_concat)

        # Task classification  Loss
        src_cls_loss = self.cross_entropy(src_pred.squeeze(), src_y)

        # conditional entropy loss.
        loss_trg_cent = self.criterion_cond(trg_pred)

        # total loss
        loss = self.hparams["src_cls_loss_wt"] * src_cls_loss + self.hparams["domain_loss_wt"] * domain_loss + \
               self.hparams["cond_ent_wt"] * loss_trg_cent

        # update feature extractor
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'Total_loss': loss.item(), 'Domain_loss': domain_loss.item(), 'Src_cls_loss': src_cls_loss.item(),
                'cond_ent_loss': loss_trg_cent.item()}


class DIRT(Algorithm):
    """
    DIRT-T: https://arxiv.org/abs/1802.08735
    """

    def __init__(self, backbone_fe, configs, hparams, device):
        super(DIRT, self).__init__(configs)

        self.feature_extractor = backbone_fe(configs)
        self.classifier = classifier(configs)
        self.network = nn.Sequential(self.feature_extractor, self.classifier)

        self.domain_classifier = Discriminator(configs)

        # optimizers
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )
        self.optimizer_disc = torch.optim.Adam(
            self.domain_classifier.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )
        # hparams
        self.hparams = hparams

        # criterion
        self.criterion_cond = ConditionalEntropyLoss().to(device)
        self.vat_loss = VAT(self.network, device).to(device)

        # device for further usage
        self.device = device

        self.ema = EMA(0.998)
        self.ema.register(self.network)

    def update(self, src_x, src_y, trg_x):
        # prepare true domain labels
        domain_label_src = torch.ones(len(src_x)).to(self.device)
        domain_label_trg = torch.zeros(len(trg_x)).to(self.device)
        domain_label_concat = torch.cat((domain_label_src, domain_label_trg), 0).long()

        src_feat = self.feature_extractor(src_x)
        src_pred = self.classifier(src_feat)

        # target features and predictions
        trg_feat = self.feature_extractor(trg_x)
        trg_pred = self.classifier(trg_feat)

        # concatenate features and predictions
        feat_concat = torch.cat((src_feat, trg_feat), dim=0)

        # Domain classification loss
        disc_prediction = self.domain_classifier(feat_concat.detach())
        disc_loss = self.cross_entropy(disc_prediction, domain_label_concat)

        # update Domain classification
        self.optimizer_disc.zero_grad()
        disc_loss.backward()
        self.optimizer_disc.step()

        # prepare fake domain labels for training the feature extractor
        domain_label_src = torch.zeros(len(src_x)).long().to(self.device)
        domain_label_trg = torch.ones(len(trg_x)).long().to(self.device)
        domain_label_concat = torch.cat((domain_label_src, domain_label_trg), 0)

        # Repeat predictions after updating discriminator
        disc_prediction = self.domain_classifier(feat_concat)

        # loss of domain discriminator according to fake labels
        domain_loss = self.cross_entropy(disc_prediction, domain_label_concat)

        # Task classification  Loss
        src_cls_loss = self.cross_entropy(src_pred.squeeze(), src_y)

        # conditional entropy loss.
        loss_trg_cent = self.criterion_cond(trg_pred)

        # Virual advariarial training loss
        loss_src_vat = self.vat_loss(src_x, src_pred)
        loss_trg_vat = self.vat_loss(trg_x, trg_pred)
        total_vat = loss_src_vat + loss_trg_vat
        # total loss
        loss = self.hparams["src_cls_loss_wt"] * src_cls_loss + self.hparams["domain_loss_wt"] * domain_loss + \
               self.hparams["cond_ent_wt"] * loss_trg_cent + self.hparams["vat_loss_wt"] * total_vat

        # update exponential moving average
        self.ema(self.network)

        # update feature extractor
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'Total_loss': loss.item(), 'Domain_loss': domain_loss.item(), 'Src_cls_loss': src_cls_loss.item(),
                'cond_ent_loss': loss_trg_cent.item()}


class DSAN(Algorithm):
    """
    DSAN: https://ieeexplore.ieee.org/document/9085896
    """

    def __init__(self, backbone_fe, configs, hparams, device):
        super(DSAN, self).__init__(configs)

        self.feature_extractor = backbone_fe(configs)
        self.classifier = classifier(configs)
        self.network = nn.Sequential(self.feature_extractor, self.classifier)

        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )
        self.hparams = hparams
        self.device = device
        self.loss_LMMD = LMMD_loss(device=device, class_num=configs.num_classes).to(device)

    def update(self, src_x, src_y, trg_x):
        # extract source features
        src_feat = self.feature_extractor(src_x)
        src_pred = self.classifier(src_feat)

        # extract target features
        trg_feat = self.feature_extractor(trg_x)
        trg_pred = self.classifier(trg_feat)

        # calculate lmmd loss
        domain_loss = self.loss_LMMD.get_loss(src_feat, trg_feat, src_y, torch.nn.functional.softmax(trg_pred, dim=1))

        # calculate source classification loss
        src_cls_loss = self.cross_entropy(src_pred, src_y)

        # calculate the total loss
        loss = self.hparams["domain_loss_wt"] * domain_loss + \
               self.hparams["src_cls_loss_wt"] * src_cls_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'Total_loss': loss.item(), 'LMMD_loss': domain_loss.item(), 'Src_cls_loss': src_cls_loss.item()}


class HoMM(Algorithm):
    """
    HoMM: https://arxiv.org/pdf/1912.11976.pdf
    """

    def __init__(self, backbone_fe, configs, hparams, device):
        super(HoMM, self).__init__(configs)

        self.coral = CORAL()

        self.feature_extractor = backbone_fe(configs)
        self.classifier = classifier(configs)
        self.network = nn.Sequential(self.feature_extractor, self.classifier)

        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )
        self.hparams = hparams
        self.device = device
        self.HoMM_loss = HoMM_loss()

    def update(self, src_x, src_y, trg_x):
        # extract source features
        src_feat = self.feature_extractor(src_x)
        src_pred = self.classifier(src_feat)

        # extract target features
        trg_feat = self.feature_extractor(trg_x)
        trg_pred = self.classifier(trg_feat)

        # calculate source classification loss
        src_cls_loss = self.cross_entropy(src_pred, src_y)

        # calculate lmmd loss
        domain_loss = self.HoMM_loss(src_feat, trg_feat)

        # calculate the total loss
        loss = self.hparams["domain_loss_wt"] * domain_loss + \
               self.hparams["src_cls_loss_wt"] * src_cls_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'Total_loss': loss.item(), 'HoMM_loss': domain_loss.item(), 'Src_cls_loss': src_cls_loss.item()}


class DDC(Algorithm):
    """
    DDC: https://arxiv.org/abs/1412.3474
    """

    def __init__(self, backbone_fe, configs, hparams, device):
        super(DDC, self).__init__(configs)

        self.feature_extractor = backbone_fe(configs)
        self.classifier = classifier(configs)
        self.network = nn.Sequential(self.feature_extractor, self.classifier)

        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )
        self.hparams = hparams
        self.device = device
        self.mmd_loss = MMD_loss()

    def update(self, src_x, src_y, trg_x):
        # extract source features
        src_feat = self.feature_extractor(src_x)
        src_pred = self.classifier(src_feat)

        # extract target features
        trg_feat = self.feature_extractor(trg_x)

        # calculate source classification loss
        src_cls_loss = self.cross_entropy(src_pred, src_y)

        # calculate mmd loss
        domain_loss = self.mmd_loss(src_feat, trg_feat)

        # calculate the total loss
        loss = self.hparams["domain_loss_wt"] * domain_loss + \
               self.hparams["src_cls_loss_wt"] * src_cls_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'Total_loss': loss.item(), 'MMD_loss': domain_loss.item(), 'Src_cls_loss': src_cls_loss.item()}


class CoDATS(Algorithm):
    """
    CoDATS: https://arxiv.org/pdf/2005.10996.pdf
    """

    def __init__(self, backbone_fe, configs, hparams, device):
        super(CoDATS, self).__init__(configs)

        self.feature_extractor = backbone_fe(configs)
        # we replace the original classifier with codats the classifier
        # remember to use same name of self.classifier, as we use it for the model evaluation
        self.classifier = codats_classifier(configs)
        self.network = nn.Sequential(self.feature_extractor, self.classifier)

        self.domain_classifier = Discriminator(configs)

        self.optimizer = torch.optim.Adam(
            list(self.feature_extractor.parameters()) + list(self.classifier.parameters()),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"], betas=(0.5, 0.99)
        )
        self.optimizer_disc = torch.optim.Adam(
            self.domain_classifier.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"], betas=(0.5, 0.99)
        )
        self.hparams = hparams
        self.device = device

    def update(self, src_x, src_y, trg_x, step, epoch, len_dataloader):
        p = float(step + epoch * len_dataloader) / self.hparams["num_epochs"] + 1 / len_dataloader
        alpha = 2. / (1. + np.exp(-10 * p)) - 1

        # zero grad
        self.optimizer.zero_grad()
        self.optimizer_disc.zero_grad()

        domain_label_src = torch.ones(len(src_x)).to(self.device)
        domain_label_trg = torch.zeros(len(trg_x)).to(self.device)

        src_feat = self.feature_extractor(src_x)
        src_pred = self.classifier(src_feat)

        trg_feat = self.feature_extractor(trg_x)

        # Task classification  Loss
        src_cls_loss = self.cross_entropy(src_pred.squeeze(), src_y)

        # Domain classification loss
        # source
        src_feat_reversed = ReverseLayerF.apply(src_feat, alpha)
        src_domain_pred = self.domain_classifier(src_feat_reversed)
        src_domain_loss = self.cross_entropy(src_domain_pred, domain_label_src.long())

        # target
        trg_feat_reversed = ReverseLayerF.apply(trg_feat, alpha)
        trg_domain_pred = self.domain_classifier(trg_feat_reversed)
        trg_domain_loss = self.cross_entropy(trg_domain_pred, domain_label_trg.long())

        # Total domain loss
        domain_loss = src_domain_loss + trg_domain_loss

        loss = self.hparams["src_cls_loss_wt"] * src_cls_loss + \
               self.hparams["domain_loss_wt"] * domain_loss

        loss.backward()
        self.optimizer.step()
        self.optimizer_disc.step()

        return {'Total_loss': loss.item(), 'Domain_loss': domain_loss.item(), 'Src_cls_loss': src_cls_loss.item()}


class AdvSKM(Algorithm):
    """
    AdvSKM: https://www.ijcai.org/proceedings/2021/0378.pdf
    """

    def __init__(self, backbone_fe, configs, hparams, device):
        super(AdvSKM, self).__init__(configs)
        self.AdvSKM_embedder = AdvSKM_Disc(configs).to(device)

        self.feature_extractor = backbone_fe(configs)
        self.classifier = classifier(configs)
        self.network = nn.Sequential(self.feature_extractor, self.classifier)

        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )

        self.optimizer_disc = torch.optim.Adam(
            self.AdvSKM_embedder.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )

        self.hparams = hparams
        self.device = device
        self.mmd_loss = MMD_loss()

    def update(self, src_x, src_y, trg_x):
        # extract source features
        src_feat = self.feature_extractor(src_x)
        src_pred = self.classifier(src_feat)

        # extract target features
        trg_feat = self.feature_extractor(trg_x)

        source_embedding_disc = self.AdvSKM_embedder(src_feat.detach())
        target_embedding_disc = self.AdvSKM_embedder(trg_feat.detach())
        mmd_loss = - self.mmd_loss(source_embedding_disc, target_embedding_disc)
        mmd_loss.requires_grad = True

        # update discriminator
        self.optimizer_disc.zero_grad()
        mmd_loss.backward()
        self.optimizer_disc.step()

        # calculate source classification loss
        src_cls_loss = self.cross_entropy(src_pred, src_y)

        # domain loss.
        source_embedding_disc = self.AdvSKM_embedder(src_feat)
        target_embedding_disc = self.AdvSKM_embedder(trg_feat)

        mmd_loss_adv = self.mmd_loss(source_embedding_disc, target_embedding_disc)
        mmd_loss_adv.requires_grad = True

        # calculate the total loss
        loss = self.hparams["domain_loss_wt"] * mmd_loss_adv + \
               self.hparams["src_cls_loss_wt"] * src_cls_loss

        # update optimizer
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'Total_loss': loss.item(), 'MMD_loss': mmd_loss_adv.item(), 'Src_cls_loss': src_cls_loss.item()}


class CDKD(Algorithm):
    """
    CDKD
    """

    def __init__(self, backbone_fe, configs, hparams, device):
        super(CDKD, self).__init__(configs)
        from models import models
        self.t_feature_extractor = models.CNN_T(configs)
        self.t_classifier = models.classifier_T(configs)
        self.network_t = nn.Sequential(self.t_feature_extractor, self.t_classifier)

        self.feature_extractor = backbone_fe(configs)
        self.classifier = classifier(configs)
        self.network = nn.Sequential(self.feature_extractor, self.classifier)

        self.domain_classifier = Discriminator(configs)

        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"], betas=(0.5, 0.99)
        )
        self.optimizer_disc = torch.optim.Adam(
            self.domain_classifier.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"], betas=(0.5, 0.99)
        )
        self.hparams = hparams
        self.device = device
        self.temperature = hparams["temperature"]

    def cosine_similarity_loss(self, output_net, target_net, eps=0.0000001):
        # Normalize each vector by its norm
        output_net_norm = torch.sqrt(torch.sum(output_net ** 2, dim=1, keepdim=True))
        output_net = output_net / (output_net_norm + eps)
        output_net[output_net != output_net] = 0

        target_net_norm = torch.sqrt(torch.sum(target_net ** 2, dim=1, keepdim=True))
        target_net = target_net / (target_net_norm + eps)
        target_net[target_net != target_net] = 0

        # Calculate the cosine similarity
        model_similarity = torch.mm(output_net, output_net.transpose(0, 1))
        target_similarity = torch.mm(target_net, target_net.transpose(0, 1))

        # Scale cosine similarity to 0..1
        model_similarity = (model_similarity + 1.0) / 2.0
        target_similarity = (target_similarity + 1.0) / 2.0

        # Transform them into probabilities
        model_similarity = model_similarity / torch.sum(model_similarity, dim=1, keepdim=True)
        target_similarity = target_similarity / torch.sum(target_similarity, dim=1, keepdim=True)

        # Calculate the KL-divergence
        loss = torch.mean(target_similarity * torch.log((target_similarity + eps) / (model_similarity + eps)))

        return loss


    def update(self, src_x, src_y, trg_x, step, epoch, len_dataloader):
        p = float(step + epoch * len_dataloader) / self.hparams["num_epochs"] + 1 / len_dataloader
        alpha = 2. / (1. + np.exp(-10 * p)) - 1

        # zero grad
        self.optimizer.zero_grad()
        self.optimizer_disc.zero_grad()

        self.network_t.eval()

        domain_label_src = torch.ones(len(src_x)).to(self.device)
        domain_label_trg = torch.zeros(len(trg_x)).to(self.device)

        src_feat_t = self.t_feature_extractor(src_x)
        src_pred_t = self.t_classifier(src_feat_t)
        src_pred_t_soften = torch.nn.functional.log_softmax(src_pred_t/self.temperature,dim=1)

        src_feat = self.feature_extractor(src_x)
        src_pred = self.classifier(src_feat)
        src_pred_s_soften = torch.nn.functional.log_softmax(src_pred / self.temperature, dim=1)

        trg_pred_t = self.t_classifier(src_feat_t)
        trg_pred_t_soften = torch.nn.functional.log_softmax(trg_pred_t / self.temperature, dim=1)
        trg_pred = self.classifier(src_feat)
        trg_pred_s_soften = torch.nn.functional.log_softmax(trg_pred / self.temperature, dim=1)

        trg_feat = self.feature_extractor(trg_x)

        # Task classification  Loss
        src_cls_loss = self.cross_entropy(src_pred.squeeze(), src_y)

        # Domain classification loss
        # source
        src_feat_reversed = ReverseLayerF.apply(src_feat, alpha)
        src_domain_pred = self.domain_classifier(src_feat_reversed)
        src_domain_loss = self.cross_entropy(src_domain_pred, domain_label_src.long())

        src_dis_pred_t = torch.nn.functional.softmax(src_domain_pred, dim=1)
        weights_src = 1 - torch.abs(src_dis_pred_t[:, 0] - src_dis_pred_t[:, 1])

        # target
        trg_feat_reversed = ReverseLayerF.apply(trg_feat, alpha)
        trg_domain_pred = self.domain_classifier(trg_feat_reversed)
        trg_domain_loss = self.cross_entropy(trg_domain_pred, domain_label_trg.long())

        trg_dis_pred_t = torch.nn.functional.softmax(trg_domain_pred, dim=1)
        weights_trg = 1 - torch.abs(trg_dis_pred_t[:, 0] - trg_dis_pred_t[:, 1])

        # Total domain loss
        domain_loss = src_domain_loss + trg_domain_loss

        # KD loss
        # soft_loss = torch.nn.functional.kl_div(src_pred_s_soften, src_pred_t_soften, reduction='batchmean', log_target=True)

        soft_loss_skd = torch.nn.functional.softmax(src_pred_t / self.temperature, dim=1) * \
                        (torch.log(torch.nn.functional.softmax(src_pred_t / self.temperature, dim=1))
                         - torch.nn.functional.log_softmax(src_pred / self.temperature, dim=1))
        # soft_loss_skd = soft_loss_skd.sum()/src_pred.size(0) # Original Implementation
        soft_loss_skd = (soft_loss_skd.sum(dim=1) * weights_src).sum(dim=0) / src_pred.size(0)

        soft_loss_tkd = torch.nn.functional.softmax(trg_pred_t / self.temperature, dim=1) * \
                        (torch.log(torch.nn.functional.softmax(trg_pred_t / self.temperature, dim=1))
                         - torch.nn.functional.log_softmax(trg_pred / self.temperature, dim=1))
        # soft_loss_tkd = soft_loss_tkd.sum()/trg_pred.size(0) # Original Implementation
        soft_loss_tkd = (soft_loss_tkd.sum(dim=1) * weights_trg).sum(dim=0) / trg_pred.size(0)
        soft_loss = soft_loss_skd + soft_loss_tkd

        kd_loss = soft_loss * self.temperature ** 2

        # KD feature loss
        # fea_loss = self.cosine_similarity_loss(src_feat,src_feat_t )
        # fea_loss = torch.nn.functional.mse_loss(src_feat,src_feat_t)

        # loss = self.hparams["src_cls_loss_wt"] * src_cls_loss + self.hparams["domain_loss_wt"] * domain_loss + \
        #        self.hparams["soft_loss_wt"] * kd_loss

        # loss = self.hparams["src_cls_loss_wt"] * src_cls_loss + self.hparams["domain_loss_wt"] *  domain_loss + \
        #        self.hparams["soft_loss_wt"] *kd_loss + fea_loss

        import math
        g = math.log10(0.9 / 0.1) / self.hparams["num_epochs"]
        beta = 0.1 * math.exp(g * epoch)
        loss = self.hparams["src_cls_loss_wt"] * src_cls_loss + self.hparams["domain_loss_wt"] *  domain_loss + \
               beta*kd_loss

        loss.backward()
        self.optimizer.step()
        self.optimizer_disc.step()

        # return {'Total_loss': loss.item(), 'Domain_loss': domain_loss.item(), 'Src_cls_loss': src_cls_loss.item(), 'kd_loss':kd_loss.item()}

        return {'Total_loss': loss.item(), 'Domain_loss': domain_loss.item(), 'Src_cls_loss': src_cls_loss.item(),
            'kd_loss': kd_loss.item()}


class AAD(Algorithm):
    """
    AAD
    """

    def __init__(self, backbone_fe, configs, hparams, device):
        super(AAD, self).__init__(configs)
        from models import models
        self.t_feature_extractor = models.CNN_T(configs)
        self.t_classifier = models.classifier_T(configs)
        self.network_t = nn.Sequential(self.t_feature_extractor, self.t_classifier)

        self.feature_extractor = backbone_fe(configs)
        self.classifier = classifier(configs)
        self.network = nn.Sequential(self.feature_extractor, self.classifier)

        self.data_domain_classifier = Discriminator_t(configs)
        self.feature_domain_classifier = Discriminator_fea(configs)
        self.adapter = Adapter(configs)

        self.optimizer = torch.optim.Adam(
            list(self.network.parameters()) + list(self.adapter.parameters()),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"], betas=(0.5, 0.99)
        )

        self.optimizer_disc = torch.optim.Adam(
            self.data_domain_classifier.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"], betas=(0.5, 0.99)
        )

        self.optimizer_feat = torch.optim.Adam(
            self.feature_domain_classifier.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"], betas=(0.5, 0.99)
        )

        self.hparams = hparams
        self.device = device
        self.temperature = hparams["temperature"]


    def update(self, src_x, src_y, trg_x, step, epoch, len_dataloader):
        p = float(step + epoch * len_dataloader) / self.hparams["num_epochs"] + 1 / len_dataloader
        alpha = 2. / (1. + np.exp(-10 * p)) - 1
        self.network_t.eval()

        real_label = 1
        fake_label = 0

        ########################################################
        # (1) update D network: maximize log(D(fea_t)) + log(1-D(G(x))
        # fea_t: feature from teacher network
        # x: input data
        # G(x): feature from student network
        ########################################################
        self.optimizer_feat.zero_grad()

        # Format Batch
        src_feat_t = self.t_feature_extractor(src_x)
        src_feat_t = Variable(src_feat_t, requires_grad=False)

        f_domain_label = torch.full((src_x.shape[0],), real_label, dtype=torch.float, device=self.device)

        # Forward pass real batch through D
        output = self.feature_domain_classifier(src_feat_t).view(-1)
        # Calculate loss on all-real batch
        errD_real = nn.BCELoss()(output, f_domain_label)
        # Calculate gradients for D in backward pass
        errD_real.backward()

        # Train with all-fake batch, Generate fake features with G
        # Student Forward
        src_feat = self.feature_extractor(src_x)
        src_feat_hint = self.adapter(src_feat)

        f_domain_label.fill_(fake_label)
        # Classify all fake batch with D

        output = self.feature_domain_classifier(src_feat_hint.detach()).view(-1)
        # output = self.feature_domain_classifier(src_feat.detach()).view(-1)
        # Calculate D's loss on the all-fake batch
        errD_fake = nn.BCELoss()(output, f_domain_label)
        # Calculate the gradients for this batch
        errD_fake.backward()

        # Add the gradients from all-real and all-fake batches
        errD = errD_real + errD_fake
        # Update D
        self.optimizer_feat.step()

        ########################################################
        # (2) update G network: maximize log(D(G(x))
        ########################################################

        # zero grad
        self.optimizer.zero_grad()
        self.optimizer_disc.zero_grad()

        src_pred_t = self.t_classifier(src_feat_t)
        src_pred_t_soften = torch.nn.functional.log_softmax(src_pred_t / self.temperature, dim=1)


        src_pred = self.classifier(src_feat)
        src_pred_s_soften = torch.nn.functional.log_softmax(src_pred / self.temperature, dim=1)

        # fake labels are real for generator cost
        f_domain_label.fill_(real_label)
        # Since we just updated D, perform another forward pass of all-fake batch through D
        output = self.feature_domain_classifier(src_feat_hint).view(-1)

        # Calculate G's loss based on this output
        errG = nn.BCELoss()(output, f_domain_label)

        # Add KD loss
        soft_loss = torch.nn.functional.kl_div(src_pred_s_soften, src_pred_t_soften, reduction='batchmean', log_target=True)
        kd_loss = soft_loss * self.temperature ** 2

        # Task classification  Loss
        src_cls_loss = self.cross_entropy(src_pred.squeeze(), src_y)


        loss = self.hparams["src_cls_loss_wt"] * src_cls_loss + self.hparams["soft_loss_wt"] * kd_loss + self.hparams ['errG'] * errG

        loss.backward()
        self.optimizer.step()
        self.optimizer_disc.step()

        return {'Total_loss': loss.item(), 'Src_cls_loss': src_cls_loss.item(), 'KD_loss':kd_loss.item(), 'errD': errD.item(), 'errG':errG.item() }


class AdvCDKDv2(Algorithm):
    """
    AdvCDKD
    """

    def __init__(self, backbone_fe, configs, hparams, device):
        super(AdvCDKDv2, self).__init__(configs)
        from models import models
        self.t_feature_extractor = models.CNN_T(configs)
        self.t_classifier = models.classifier_T(configs)
        self.network_t = nn.Sequential(self.t_feature_extractor, self.t_classifier)

        self.feature_extractor = backbone_fe(configs)
        self.classifier = classifier(configs)
        self.network = nn.Sequential(self.feature_extractor, self.classifier)

        self.data_domain_classifier_t = Discriminator_t(configs)

        self.data_domain_classifier = Discriminator(configs)
        self.feature_domain_classifier = Discriminator_fea(configs)
        self.adapter = Adapter(configs)

        self.optimizer = torch.optim.Adam(
            list(self.network.parameters()) + list(self.adapter.parameters()),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"], betas=(0.5, 0.99)
        )

        self.optimizer_disc = torch.optim.Adam(
            self.data_domain_classifier.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"], betas=(0.5, 0.99)
        )

        self.optimizer_feat = torch.optim.Adam(
            self.feature_domain_classifier.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"], betas=(0.5, 0.99)
        )

        self.hparams = hparams
        self.device = device
        self.temperature = hparams["temperature"]


    def update(self, src_x, src_y, trg_x, step, epoch, len_dataloader):
        p = float(step + epoch * len_dataloader) / self.hparams["num_epochs"] + 1 / len_dataloader
        alpha = 2. / (1. + np.exp(-10 * p)) - 1
        self.network_t.eval()

        real_label = 1
        fake_label = 0

        ########################################################
        # (1) update D network: maximize log(D(fea_t)) + log(1-D(G(x))
        # fea_t: feature from teacher network
        # x: input data
        # G(x): feature from student network
        ########################################################
        self.optimizer_feat.zero_grad()

        # Format Batch
        src_feat_t = self.t_feature_extractor(src_x)
        # src_dis_pred_t = torch.nn.functional.softmax(self.data_domain_classifier_t(src_feat_t),dim=1)
        # weights_src = 1 - torch.abs(src_dis_pred_t[:,0]-src_dis_pred_t[:,1])
        src_feat_t = Variable(src_feat_t, requires_grad=False)

        trg_feat_t = self.t_feature_extractor(trg_x)
        # trg_dis_pred_t = torch.nn.functional.softmax(self.data_domain_classifier_t(trg_feat_t),dim=1)
        # weights_trg = 1 - torch.abs(trg_dis_pred_t[:, 0] - trg_dis_pred_t[:, 1])
        trg_feat_t = Variable(trg_feat_t, requires_grad=False)

        f_domain_label = torch.full((src_x.shape[0]+trg_x.shape[0],), real_label, dtype=torch.float, device=self.device)

        # Forward pass real batch through D
        output = self.feature_domain_classifier(torch.concat((src_feat_t,trg_feat_t),dim=0)).view(-1)
        # Calculate loss on all-real batch
        errD_real = nn.BCELoss()(output, f_domain_label)
        # Calculate gradients for D in backward pass
        errD_real.backward()

        # Train with all-fake batch, Generate fake features with G
        # Student Forward
        src_feat = self.feature_extractor(src_x)
        src_feat_hint = self.adapter(src_feat)

        trg_feat = self.feature_extractor(trg_x)
        trg_feat_hint = self.adapter(trg_feat)

        f_domain_label.fill_(fake_label)
        # Classify all fake batch with D

        fea_hint = torch.concat((src_feat_hint,trg_feat_hint),dim=0).detach()

        output = self.feature_domain_classifier(fea_hint).view(-1)
        # output = self.feature_domain_classifier(src_feat.detach()).view(-1)
        # Calculate D's loss on the all-fake batch
        errD_fake = nn.BCELoss()(output, f_domain_label)
        # Calculate the gradients for this batch
        errD_fake.backward()

        # Add the gradients from all-real and all-fake batches
        errD = errD_real + errD_fake
        # Update D
        self.optimizer_feat.step()

        ########################################################
        # (2) update G network: maximize log(D(G(x))
        ########################################################

        # zero grad
        self.optimizer.zero_grad()
        self.optimizer_disc.zero_grad()

        src_pred_t = self.t_classifier(src_feat_t)
        src_pred_t_soften = torch.nn.functional.log_softmax(src_pred_t / self.temperature, dim=1)
        src_pred = self.classifier(src_feat)
        src_pred_s_soften = torch.nn.functional.log_softmax(src_pred / self.temperature, dim=1)

        trg_pred_t = self.t_classifier(src_feat_t)
        trg_pred_t_soften = torch.nn.functional.log_softmax(trg_pred_t / self.temperature, dim=1)
        trg_pred = self.classifier(src_feat)
        trg_pred_s_soften = torch.nn.functional.log_softmax(trg_pred / self.temperature, dim=1)

        # fake labels are real for generator cost
        f_domain_label.fill_(real_label)
        # Since we just updated D, perform another forward pass of all-fake batch through D
        output = self.feature_domain_classifier(fea_hint).view(-1)
        # output = self.feature_domain_classifier(src_feat).view(-1)

        # Calculate G's loss based on this output
        errG = nn.BCELoss()(output, f_domain_label)

        # Add L1 loss to optimize adapter layer
        errL1 = nn.L1Loss()(src_feat_hint, src_feat_t) + nn.L1Loss()(trg_feat_hint,trg_feat_t)

        errG = errG + errL1

        # Domain classification loss
        domain_label_src = torch.ones(len(src_x)).to(self.device)
        domain_label_trg = torch.zeros(len(trg_x)).to(self.device)

        trg_feat = self.feature_extractor(trg_x)

        # source
        src_feat_reversed = ReverseLayerF.apply(src_feat, alpha)
        src_domain_pred = self.data_domain_classifier(src_feat_reversed)
        src_domain_loss = self.cross_entropy(src_domain_pred, domain_label_src.long())

        src_dis_pred_t = torch.nn.functional.softmax(src_domain_pred, dim=1)
        weights_src = 1 - torch.abs(src_dis_pred_t[:, 0] - src_dis_pred_t[:, 1])

        # target
        trg_feat_reversed = ReverseLayerF.apply(trg_feat, alpha)
        trg_domain_pred = self.data_domain_classifier(trg_feat_reversed)
        trg_domain_loss = self.cross_entropy(trg_domain_pred, domain_label_trg.long())

        trg_dis_pred_t = torch.nn.functional.softmax(trg_domain_pred,dim=1)
        weights_trg = 1 - torch.abs(trg_dis_pred_t[:, 0] - trg_dis_pred_t[:, 1])

        # Total domain loss
        domain_loss = src_domain_loss + trg_domain_loss

        # Add KD loss

        # Normal Knowledge
        # soft_loss = torch.nn.functional.kl_div(src_pred_s_soften, src_pred_t_soften, reduction='batchmean', log_target=True)\
        #             + torch.nn.functional.kl_div(trg_pred_s_soften, trg_pred_t_soften, reduction='batchmean', log_target=True)

        # Disentangled Knowledge
        soft_loss_skd = torch.nn.functional.softmax(src_pred_t/self.temperature, dim=1) * \
                     (torch.log(torch.nn.functional.softmax(src_pred_t/self.temperature, dim=1))
                      - torch.nn.functional.log_softmax(src_pred/self.temperature, dim=1))
        # soft_loss_skd = soft_loss_skd.sum()/src_pred.size(0) # Original Implementation
        soft_loss_skd = (soft_loss_skd.sum(dim=1) * weights_src).sum(dim=0) / src_pred.size(0)

        soft_loss_tkd = torch.nn.functional.softmax(trg_pred_t/self.temperature, dim=1) * \
                     (torch.log(torch.nn.functional.softmax(trg_pred_t/self.temperature, dim=1))
                      - torch.nn.functional.log_softmax(trg_pred/self.temperature, dim=1))
        # soft_loss_tkd = soft_loss_tkd.sum()/trg_pred.size(0) # Original Implementation
        soft_loss_tkd = (soft_loss_tkd.sum(dim=1) * weights_trg).sum(dim=0) / trg_pred.size(0)
        soft_loss = soft_loss_skd + soft_loss_tkd

        kd_loss = soft_loss * self.temperature ** 2

        # Task classification  Loss
        src_cls_loss = self.cross_entropy(src_pred.squeeze(), src_y)

        # loss = self.hparams["src_cls_loss_wt"] * src_cls_loss + self.hparams["domain_loss_wt"] *  domain_loss + \
        #        self.hparams["soft_loss_wt"] * kd_loss + self.hparams ['errG'] * errG + errL1

        import math
        g = math.log10(0.9/0.1) / self.hparams["num_epochs"]
        beta = 0.1 * math.exp(g*epoch)

        alpha = 0.1

        # beta = 0.1

        # loss = self.hparams["src_cls_loss_wt"] * src_cls_loss + (1-beta)*self.hparams["domain_loss_wt"] *  domain_loss + \
        #        beta * kd_loss + self.hparams ['errG'] * errG

        loss = alpha * src_cls_loss + (1-beta)* self.hparams["domain_loss_wt"] * domain_loss + beta * kd_loss + errG

        # loss = self.hparams["src_cls_loss_wt"] * src_cls_loss + self.hparams["domain_loss_wt"] *  domain_loss + \
        #        beta * kd_loss + self.hparams ['errG'] * errG

        loss.backward()
        self.optimizer.step()
        self.optimizer_disc.step()

        return {'Total_loss': loss.item(), 'Domain_loss': domain_loss.item(), 'Src_cls_loss': src_cls_loss.item(),
                'KD_loss':kd_loss.item(), 'errD': errD.item(), 'errG':errG.item()}


class KDSTDA(Algorithm):
    """
    JointUKD
    """

    def __init__(self, backbone_fe, configs, hparams, device):
        super(KDSTDA, self).__init__(configs)
        from models import models
        self.t_feature_extractor = models.CNN_T(configs)
        self.t_classifier = models.classifier_T(configs)
        self.network_t = nn.Sequential(self.t_feature_extractor, self.t_classifier)

        self.feature_extractor = backbone_fe(configs)
        self.classifier = classifier(configs)
        self.network = nn.Sequential(self.feature_extractor, self.classifier)

        self.domain_classifier = Discriminator(configs)

        self.optimizer = torch.optim.Adam(
            list(self.network.parameters()) + list(self.network_t.parameters()),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"], betas=(0.5, 0.99)
        )

        self.hparams = hparams
        self.device = device
        self.temperature = hparams["temperature"]


    def update(self, src_x, src_y, trg_x, step, epoch, len_dataloader):
        p = float(step + epoch * len_dataloader) / self.hparams["num_epochs"] + 1 / len_dataloader
        alpha = 2. / (1. + np.exp(-10 * p)) - 1

        # zero grad
        self.optimizer.zero_grad()
        self.network_t.train()

        # Teacher inference on Source and Target
        src_feat_t = self.t_feature_extractor(src_x)
        src_pred_t = self.t_classifier(src_feat_t)
        src_pred_t_soften = torch.nn.functional.log_softmax(src_pred_t/self.temperature,dim=1)

        trg_feat_t = self.t_feature_extractor(trg_x)
        trg_pred_t = self.t_classifier(trg_feat_t)
        trg_pred_t_soften = torch.nn.functional.log_softmax(trg_pred_t / self.temperature, dim=1)

        # Student inference on Source and Target
        src_feat = self.feature_extractor(src_x)
        src_pred = self.classifier(src_feat)
        src_pred_s_soften = torch.nn.functional.log_softmax(src_pred / self.temperature, dim=1)

        trg_feat = self.feature_extractor(trg_x)
        trg_pred = self.classifier(trg_feat)
        trg_pred_s_soft = torch.nn.functional.log_softmax(trg_pred / self.temperature, dim=1)

        from mmd import MMD_loss
        mmd_loss = MMD_loss()(src_feat_t,trg_feat_t)
        loss_ce_t = self.cross_entropy(src_pred_t, src_y)
        loss_tda = mmd_loss + 0.8 * loss_ce_t

        loss_tkd = torch.nn.functional.kl_div(trg_pred_s_soft, trg_pred_t_soften, reduction='batchmean', log_target=True)

        loss_kd_src = torch.nn.functional.kl_div(src_pred_s_soften, src_pred_t_soften, reduction='batchmean', log_target=True)
        loss_ce_s = self.cross_entropy(src_pred, src_y)

        loss_skd = loss_kd_src + 0.8 * loss_ce_s
        import math
        g = math.log10(0.9/0.1) / self.hparams["num_epochs"]
        beta = 0.1 * math.exp(g*epoch)
        loss = (1-beta) * loss_tda + beta*(loss_skd+loss_tkd)
        loss.backward()
        self.optimizer.step()

        return {'Total_loss': loss.item(), 'loss_tda': loss_tda.item(), 'loss_skd': loss_skd.item(), 'loss_tkd':loss_tkd.item()}


class MobileDA(Algorithm):
    """
    MobileDA
    """

    def __init__(self, backbone_fe, configs, hparams, device):
        super(MobileDA, self).__init__(configs)
        from models import models
        self.t_feature_extractor = models.CNN_T(configs)
        self.t_classifier = models.classifier_T(configs)
        self.network_t = nn.Sequential(self.t_feature_extractor, self.t_classifier)

        self.feature_extractor = backbone_fe(configs)
        self.classifier = classifier(configs)
        self.network = nn.Sequential(self.feature_extractor, self.classifier)

        self.domain_classifier = Discriminator(configs)

        self.coral = CORAL()

        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"], betas=(0.5, 0.99)
        )

        self.hparams = hparams
        self.device = device
        self.temperature = hparams["temperature"]


    def update(self, src_x, src_y, trg_x, step, epoch, len_dataloader):
        p = float(step + epoch * len_dataloader) / self.hparams["num_epochs"] + 1 / len_dataloader
        alpha = 2. / (1. + np.exp(-10 * p)) - 1

        # zero grad
        self.optimizer.zero_grad()
        self.network_t.eval()

        trg_feat_t = self.t_feature_extractor(trg_x)
        trg_pred_t = self.t_classifier(trg_feat_t)
        trg_pred_t_soften = torch.nn.functional.log_softmax(trg_pred_t / self.temperature, dim=1)

        # Student inference on Source and Target
        src_feat = self.feature_extractor(src_x)
        src_pred = self.classifier(src_feat)
        src_pred_s_soften = torch.nn.functional.log_softmax(src_pred / self.temperature, dim=1)

        trg_feat = self.feature_extractor(trg_x)
        trg_pred = self.classifier(trg_feat)
        trg_pred_s_soft = torch.nn.functional.log_softmax(trg_pred / self.temperature, dim=1)

        loss_ce_s = self.cross_entropy(src_pred, src_y)
        loss_soft = torch.nn.functional.kl_div(trg_pred_s_soft, trg_pred_t_soften, reduction='batchmean',
                                              log_target=True)

        loss_dc = self.coral(src_feat, trg_feat)

        loss = loss_ce_s + 0.7* loss_soft + 0.3 * loss_dc
        loss.backward()
        self.optimizer.step()

        return {'Total_loss': loss.item(), 'loss_ce': loss_ce_s.item(), 'loss_soft': loss_soft.item(), 'loss_dc':loss_dc.item()}


class JointADKD(Algorithm): # Proposed AAAI
    """
    JointADKD
    """

    def __init__(self, backbone_fe, configs, hparams, device):
        super(JointADKD, self).__init__(configs)
        from models import models
        # Teacher Model and discriminator
        self.t_feature_extractor = models.CNN_T(configs)
        self.t_classifier = models.classifier_T(configs)
        self.network_t = nn.Sequential(self.t_feature_extractor, self.t_classifier)
        self.domain_classifier = Discriminator_t(configs)

        self.feature_extractor = backbone_fe(configs)
        self.classifier = classifier(configs)
        self.network = nn.Sequential(self.feature_extractor, self.classifier)

        self.adapter = Adapter(configs)

        self.optimizer = torch.optim.Adam(
            list(self.network.parameters()) + list(self.adapter.parameters()),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"], betas=(0.5, 0.99)
        )
        self.hparams = hparams
        self.device = device
        self.temperature = hparams["temperature"]

    def update_mcdo(self, src_x, src_y, trg_x):

        self.network_t.eval()

        # zero grad
        self.optimizer.zero_grad()
        # Format Batch through teacher model
        src_feat_t = self.t_feature_extractor(src_x)
        src_feat_t = Variable(src_feat_t, requires_grad=False)

        trg_feat_t = self.t_feature_extractor(trg_x)
        trg_pred_t = self.t_classifier(trg_feat_t)
        trg_pred_t = Variable(trg_pred_t, requires_grad=False)

        is_mc_dropout = True
        if is_mc_dropout:
            for module in self.t_feature_extractor.modules():
                if module.__class__.__name__.startswith('Dropout'):
                    module.train()
            for module in self.t_classifier.modules():
                if module.__class__.__name__.startswith('Dropout'):
                    module.train()

            forward_pass = 10
            n_samples = src_y.shape[0]
            n_classes = self.hparams["n_classes"]
            dropout_predictions = np.empty((0,n_samples,n_classes))
            softmax = nn.Softmax(dim=1)
            for i in range(forward_pass):
                predictions = np.empty((0, n_classes))
                trg_feat_t = self.t_feature_extractor(trg_x)
                trg_pred_t = self.t_classifier(trg_feat_t)
                output = softmax(trg_pred_t)
                predictions = np.vstack((predictions,output.detach().cpu().numpy()))
                dropout_predictions = np.vstack((dropout_predictions, predictions[np.newaxis,:,:]))

            mean = np.mean(dropout_predictions,axis=0)
            # variance = np.var(dropout_predictions,axis=0)
            epsilon = sys.float_info.min
            # Calculating entropy across multiple MCD forward passes
            entropy = -np.sum(mean * np.log(mean + epsilon), axis=-1)# shape (n_samples,)
            #
            weights = (1-entropy) / np.sum(1-entropy)

        # Student Forward
        trg_feat_s = self.feature_extractor(trg_x)
        trg_hint_s = self.adapter(trg_feat_s)
        trg_pred_s = self.classifier(trg_feat_s)

        # Calculate the KD loss
        trg_pred_t_soften = torch.nn.functional.log_softmax(trg_pred_t / self.temperature, dim=1)
        trg_pred_s_soften = torch.nn.functional.log_softmax(trg_pred_s / self.temperature, dim=1)

        if is_mc_dropout:
            soft_loss = torch.nn.functional.kl_div(trg_pred_s_soften, trg_pred_t_soften, reduction='none',
                                                   log_target=True).sum(axis=1)
            soft_loss = (torch.from_numpy(weights).to(self.device)*soft_loss).sum()
        else:
            soft_loss = torch.nn.functional.kl_div(trg_pred_s_soften, trg_pred_t_soften, reduction='batchmean', log_target=True)

        kd_loss = soft_loss * self.temperature ** 2

        # Calculate Domain confusion loss, reuse teacher's discriminator
        domain_label_src = torch.ones(len(src_x)).to(self.device)
        domain_label_trg = torch.zeros(len(trg_x)).to(self.device)

        src_domain_pred = self.domain_classifier(src_feat_t)
        src_domain_loss = self.cross_entropy(src_domain_pred, domain_label_src.long())
        trg_domain_pred = self.domain_classifier(trg_hint_s)
        trg_domain_loss = self.cross_entropy(trg_domain_pred, domain_label_trg.long())
        domain_loss = src_domain_loss + trg_domain_loss

        # Total loss
        loss= domain_loss + self.hparams["kd_loss_wt"] * kd_loss
        loss.backward()
        self.optimizer.step()

        return {'Total_loss': loss.item(), 'Domain_loss': domain_loss.item(), 'KD_loss': kd_loss.item()}

    def update_consist(self, src_x, src_y, trg_x):

        self.network_t.eval()

        # zero grad
        self.optimizer.zero_grad()
        # Format Batch through teacher model
        src_feat_t = self.t_feature_extractor(src_x)
        src_feat_t = Variable(src_feat_t, requires_grad=False)

        trg_feat_t = self.t_feature_extractor(trg_x)
        trg_pred_t = self.t_classifier(trg_feat_t)
        trg_pred_t = Variable(trg_pred_t, requires_grad=False)


        # Student Forward
        trg_feat_s = self.feature_extractor(trg_x)
        trg_hint_s = self.adapter(trg_feat_s)
        trg_pred_s = self.classifier(trg_feat_s)

        # Calculate consistency weight
        trg_x_1 = jitter(trg_x, device=self.device)
        trg_x_2 = jitter(trg_x, device=self.device)
        trg_t_x1 = self.network(trg_x_1)
        trg_t_x2 = self.network(trg_x_2)

        trg_t_y1 = torch.argmax(trg_t_x1, 1, keepdim=True)
        trg_t_y2 = torch.argmax(trg_t_x2, 1, keepdim=True)

        alpha = 0.1  # based on paper alpha = 0.5
        consist_weight = torch.FloatTensor(src_y.shape).to(self.device)
        consist_weight.zero_()
        consist_weight.scatter_(0, (trg_t_y1 == trg_t_y2).nonzero(as_tuple=True)[0],1)
        consist_weight.scatter_(0, (trg_t_y1 != trg_t_y2).nonzero(as_tuple=True)[0], alpha)

        # Calculate the KD loss
        trg_pred_t_soften = torch.nn.functional.log_softmax(trg_pred_t / self.temperature, dim=1)
        trg_pred_s_soften = torch.nn.functional.log_softmax(trg_pred_s / self.temperature, dim=1)

        soft_loss = torch.nn.functional.kl_div(trg_pred_s_soften, trg_pred_t_soften, reduction='batchmean',
                                               log_target=True)
        # soft_loss = torch.nn.functional.kl_div(trg_pred_s_soften, trg_pred_t_soften, reduction='none',
        #                                        log_target=True).sum(axis=1)
        # soft_loss = (consist_weight * soft_loss).sum()/consist_weight.shape[0]
        kd_loss = soft_loss * self.temperature ** 2

        # Calculate Domain confusion loss, reuse teacher's discriminator
        domain_label_src = torch.ones(len(src_x)).to(self.device)
        domain_label_trg = torch.zeros(len(trg_x)).to(self.device)

        src_domain_pred = self.domain_classifier(src_feat_t)
        src_domain_loss = self.cross_entropy(src_domain_pred, domain_label_src.long())
        trg_domain_pred = self.domain_classifier(trg_hint_s)
        trg_domain_loss = self.cross_entropy(trg_domain_pred, domain_label_trg.long())
        domain_loss = src_domain_loss + trg_domain_loss

        # Total loss
        loss= domain_loss + self.hparams["kd_loss_wt"] * kd_loss
        loss.backward()
        self.optimizer.step()

        return {'Total_loss': loss.item(), 'Domain_loss': domain_loss.item(), 'KD_loss': kd_loss.item()}

    def update_mcd(self, src_x, trg_x): # maximum cluster different

        self.network_t.eval()

        # zero grad
        self.optimizer.zero_grad()
        # Format Batch through teacher model
        src_feat_t = self.t_feature_extractor(src_x)
        src_feat_t = Variable(src_feat_t, requires_grad=False)

        trg_feat_t = self.t_feature_extractor(trg_x)
        trg_pred_t = self.t_classifier(trg_feat_t)
        trg_pred_t = Variable(trg_pred_t, requires_grad=False)

        # Student Forward
        trg_feat_s = self.feature_extractor(trg_x)
        trg_hint_s = self.adapter(trg_feat_s)
        trg_pred_s = self.classifier(trg_feat_s)

        # Calculate the KD loss
        trg_pred_t_soften = torch.nn.functional.log_softmax(trg_pred_t / self.temperature, dim=1)
        trg_pred_s_soften = torch.nn.functional.log_softmax(trg_pred_s / self.temperature, dim=1)

        soft_loss = torch.nn.functional.kl_div(trg_pred_s_soften, trg_pred_t_soften, reduction='batchmean', log_target=True)

        kd_loss = soft_loss * self.temperature ** 2

        # Calculate Domain confusion loss, reuse teacher's discriminator
        domain_label_src = torch.ones(len(src_x)).to(self.device)
        domain_label_trg = torch.zeros(len(trg_x)).to(self.device)

        src_domain_pred = self.domain_classifier(src_feat_t)
        src_domain_loss = self.cross_entropy(src_domain_pred, domain_label_src.long())
        trg_domain_pred = self.domain_classifier(trg_hint_s)
        trg_domain_loss = self.cross_entropy(trg_domain_pred, domain_label_trg.long())
        domain_loss = src_domain_loss + trg_domain_loss

        # Total loss
        loss= domain_loss + self.hparams["kd_loss_wt"] * kd_loss
        loss.backward()
        self.optimizer.step()

        return {'Total_loss': loss.item(), 'Domain_loss': domain_loss.item(), 'KD_loss': kd_loss.item()}

    def update_combined(self, src_x, trg_x):
        self.network_t.eval()

        # zero grad
        self.optimizer.zero_grad()
        # Format Batch through teacher model
        src_feat_t = self.t_feature_extractor(src_x)
        src_feat_t = Variable(src_feat_t, requires_grad=False)

        trg_feat_t = self.t_feature_extractor(trg_x)
        trg_pred_t = self.t_classifier(trg_feat_t)
        trg_pred_t = Variable(trg_pred_t, requires_grad=False)


        for module in self.t_feature_extractor.modules():
            if module.__class__.__name__.startswith('Dropout'):
                module.train()
        for module in self.t_classifier.modules():
            if module.__class__.__name__.startswith('Dropout'):
                module.train()

        forward_pass = 10
        n_samples = src_x.shape[0]
        n_classes = self.hparams["n_classes"]
        dropout_predictions = np.empty((0, n_samples, n_classes))
        softmax = nn.Softmax(dim=1)
        for i in range(forward_pass):
            predictions = np.empty((0, n_classes))
            #add data argumentation
            trg_x_1 = jitter(trg_x, device=self.device,sigma=0.01)

            trg_feat_t = self.t_feature_extractor(trg_x_1)
            trg_pred_t = self.t_classifier(trg_feat_t)
            output = softmax(trg_pred_t)
            predictions = np.vstack((predictions, output.detach().cpu().numpy()))
            dropout_predictions = np.vstack((dropout_predictions, predictions[np.newaxis, :, :]))

        mean = np.mean(dropout_predictions, axis=0) # shape (n_samples, n_classes)
        # variance = np.var(dropout_predictions,axis=0)
        epsilon = sys.float_info.min
        # Calculating entropy across multiple MCD forward passes
        entropy = -np.sum(mean * np.log(mean + epsilon), axis=-1)  # shape (n_samples,)
        #
        weights = (1 - entropy) / np.sum(1 - entropy)

        # Student Forward
        trg_feat_s = self.feature_extractor(trg_x)
        trg_hint_s = self.adapter(trg_feat_s)
        trg_pred_s = self.classifier(trg_feat_s)

        # Calculate the KD loss
        trg_pred_t_soften = torch.nn.functional.log_softmax(trg_pred_t / self.temperature, dim=1)
        trg_pred_s_soften = torch.nn.functional.log_softmax(trg_pred_s / self.temperature, dim=1)

        soft_loss = torch.nn.functional.kl_div(trg_pred_s_soften, trg_pred_t_soften, reduction='none',
                                                   log_target=True).sum(axis=1)
        soft_loss = (torch.from_numpy(weights).to(self.device) * soft_loss).sum()


        kd_loss = soft_loss * self.temperature ** 2

        # Calculate Domain confusion loss, reuse teacher's discriminator
        domain_label_src = torch.ones(len(src_x)).to(self.device)
        domain_label_trg = torch.zeros(len(trg_x)).to(self.device)

        src_domain_pred = self.domain_classifier(src_feat_t)
        src_domain_loss = self.cross_entropy(src_domain_pred, domain_label_src.long())
        trg_domain_pred = self.domain_classifier(trg_hint_s)
        trg_domain_loss = self.cross_entropy(trg_domain_pred, domain_label_trg.long())
        domain_loss = src_domain_loss + trg_domain_loss

        # Total loss
        loss = domain_loss + self.hparams["kd_loss_wt"] * kd_loss
        loss.backward()
        self.optimizer.step()

        return {'Total_loss': loss.item(), 'Domain_loss': domain_loss.item(), 'KD_loss': kd_loss.item()}
        pass

    def update_trans(self, src_x, src_y, trg_x):

        self.network_t.eval()
        softmax = nn.Softmax(dim=1)

        # zero grad
        self.optimizer.zero_grad()
        # Format Batch through teacher model
        src_feat_t = self.t_feature_extractor(src_x)
        src_feat_t = Variable(src_feat_t, requires_grad=False)

        trg_feat_t = self.t_feature_extractor(trg_x)
        trg_pred_t = self.t_classifier(trg_feat_t)
        trg_pred_t = Variable(trg_pred_t, requires_grad=False)

        # Calculate the rewards with MC Dropout uncertainty
        for module in self.t_feature_extractor.modules():
            if module.__class__.__name__.startswith('Dropout'):
                module.train()
        for module in self.t_classifier.modules():
            if module.__class__.__name__.startswith('Dropout'):
                module.train()

        forward_pass = 10
        n_samples = src_y.shape[0]
        n_classes = self.hparams["n_classes"]
        dropout_predictions_t = np.empty((0, n_samples, n_classes))
        softmax = nn.Softmax(dim=1)
        for i in range(forward_pass):
            predictions = np.empty((0, n_classes))
            trg_feat_temp = self.t_feature_extractor(trg_x)
            trg_pred_temp = self.t_classifier(trg_feat_temp)
            output = softmax(trg_pred_temp)
            predictions = np.vstack((predictions, output.detach().clone().cpu().numpy()))
            dropout_predictions_t = np.vstack((dropout_predictions_t, predictions[np.newaxis, :, :]))

        mean_t = np.mean(dropout_predictions_t, axis=0)

        # Student Forward
        trg_feat_s = self.feature_extractor(trg_x)
        trg_hint_s = self.adapter(trg_feat_s)
        trg_pred_s = self.classifier(trg_feat_s)
        trg_pred_s_duplicate = trg_pred_s.detach().clone()
        trg_pred_s_duplicate = softmax(trg_pred_s_duplicate).cpu().numpy()

        #  Calculate kl distance between teacher and student
        d = np.sum(kl_div(trg_pred_s_duplicate, mean_t), axis=-1)
        weights = d/ np.sum(d)

        # n = np.sign(np.mean(d) - d)
        # n[n==-1]=0 # weights

        # Calculate the KD loss
        trg_pred_t_soften = torch.nn.functional.log_softmax(trg_pred_t / self.temperature, dim=1)
        trg_pred_s_soften = torch.nn.functional.log_softmax(trg_pred_s / self.temperature, dim=1)


        soft_loss = torch.nn.functional.kl_div(trg_pred_s_soften, trg_pred_t_soften, reduction='none',
                                               log_target=True).sum(axis=1)
        soft_loss = (torch.from_numpy(weights).to(self.device)*soft_loss).sum()

        kd_loss = soft_loss * self.temperature ** 2

        # Calculate Domain confusion loss, reuse teacher's discriminator
        domain_label_src = torch.ones(len(src_x)).to(self.device)
        domain_label_trg = torch.zeros(len(trg_x)).to(self.device)

        src_domain_pred = self.domain_classifier(src_feat_t)
        src_domain_loss = self.cross_entropy(src_domain_pred, domain_label_src.long())
        trg_domain_pred = self.domain_classifier(trg_hint_s)
        trg_domain_loss = self.cross_entropy(trg_domain_pred, domain_label_trg.long())
        domain_loss = src_domain_loss + trg_domain_loss

        # Total loss
        loss= domain_loss + self.hparams["kd_loss_wt"] * kd_loss
        loss.backward()
        self.optimizer.step()

        return {'Total_loss': loss.item(), 'Domain_loss': domain_loss.item(), 'KD_loss': kd_loss.item()}

    def update_mcdo_trans(self, src_x, src_y, trg_x):

        self.network_t.eval()

        # zero grad
        self.optimizer.zero_grad()
        # Format Batch through teacher model
        src_feat_t = self.t_feature_extractor(src_x)
        src_feat_t = Variable(src_feat_t, requires_grad=False)

        trg_feat_t = self.t_feature_extractor(trg_x)
        trg_pred_t = self.t_classifier(trg_feat_t)
        trg_pred_t = Variable(trg_pred_t, requires_grad=False)

        is_mc_dropout = True
        if is_mc_dropout:
            for module in self.t_feature_extractor.modules():
                if module.__class__.__name__.startswith('Dropout'):
                    module.train()
            for module in self.t_classifier.modules():
                if module.__class__.__name__.startswith('Dropout'):
                    module.train()

            forward_pass = 10
            n_samples = src_y.shape[0]
            n_classes = self.hparams["n_classes"]
            dropout_predictions = np.empty((0,n_samples,n_classes))
            softmax = nn.Softmax(dim=1)
            for i in range(forward_pass):
                predictions = np.empty((0, n_classes))
                trg_feat_t = self.t_feature_extractor(trg_x)
                trg_pred_t = self.t_classifier(trg_feat_t)
                output = softmax(trg_pred_t)
                predictions = np.vstack((predictions,output.detach().cpu().numpy()))
                dropout_predictions = np.vstack((dropout_predictions, predictions[np.newaxis,:,:]))

            mean = np.mean(dropout_predictions,axis=0)
            # variance = np.var(dropout_predictions,axis=0)
            epsilon = sys.float_info.min
            # Calculating entropy across multiple MCD forward passes
            entropy = -np.sum(mean * np.log(mean + epsilon), axis=-1)# shape (n_samples,)
            #
            weights_mcdo = (1-entropy) / np.sum(1-entropy)

        # Student Forward
        trg_feat_s = self.feature_extractor(trg_x)
        trg_hint_s = self.adapter(trg_feat_s)
        trg_pred_s = self.classifier(trg_feat_s)

        trg_pred_s_duplicate = trg_pred_s.detach().clone()
        trg_pred_s_duplicate = softmax(trg_pred_s_duplicate).cpu().numpy()

        #  Calculate kl distance between teacher and student
        d = np.sum(kl_div(trg_pred_s_duplicate, mean), axis=-1)
        weights_trans = d/ np.sum(d)

        weights = (weights_mcdo + weights_trans)/2


        # Calculate the KD loss
        trg_pred_t_soften = torch.nn.functional.log_softmax(trg_pred_t / self.temperature, dim=1)
        trg_pred_s_soften = torch.nn.functional.log_softmax(trg_pred_s / self.temperature, dim=1)

        if is_mc_dropout:
            soft_loss = torch.nn.functional.kl_div(trg_pred_s_soften, trg_pred_t_soften, reduction='none',
                                                   log_target=True).sum(axis=1)
            soft_loss = (torch.from_numpy(weights).to(self.device)*soft_loss).sum()
        else:
            soft_loss = torch.nn.functional.kl_div(trg_pred_s_soften, trg_pred_t_soften, reduction='batchmean', log_target=True)

        kd_loss = soft_loss * self.temperature ** 2

        # Calculate Domain confusion loss, reuse teacher's discriminator
        domain_label_src = torch.ones(len(src_x)).to(self.device)
        domain_label_trg = torch.zeros(len(trg_x)).to(self.device)

        src_domain_pred = self.domain_classifier(src_feat_t)
        src_domain_loss = self.cross_entropy(src_domain_pred, domain_label_src.long())
        trg_domain_pred = self.domain_classifier(trg_hint_s)
        trg_domain_loss = self.cross_entropy(trg_domain_pred, domain_label_trg.long())
        domain_loss = src_domain_loss + trg_domain_loss

        # Total loss
        loss= domain_loss + self.hparams["kd_loss_wt"] * kd_loss
        loss.backward()
        self.optimizer.step()

        return {'Total_loss': loss.item(), 'Domain_loss': domain_loss.item(), 'KD_loss': kd_loss.item()}


class RL_JointADKD(Algorithm): # Proposed IJCAI-2024
    """
    RL-based JointADKD
    """

    def __init__(self, backbone_fe, configs, hparams, device):
        super(RL_JointADKD, self).__init__(configs)
        from models import models
        # Teacher Model and discriminator
        self.t_feature_extractor = models.CNN_T(configs)
        self.t_classifier = models.classifier_T(configs)
        self.network_t = nn.Sequential(self.t_feature_extractor, self.t_classifier)
        self.domain_classifier = Discriminator_t(configs)

        self.feature_extractor = backbone_fe(configs)
        self.classifier = classifier(configs)
        self.network = nn.Sequential(self.feature_extractor, self.classifier)

        self.feature_extractor_backup = backbone_fe(configs)
        self.classifier_backup = classifier(configs)
        self.network_backup = nn.Sequential(self.feature_extractor_backup, self.classifier_backup)

        self.adapter = Adapter(configs)

        self.Qnet = models.Qnet()
        self.Qnet_target = models.Qnet()
        self.Qnet_target.load_state_dict(self.Qnet.state_dict())
        self.episodes = hparams["episode"]
        self.qqdn_lr = hparams["ddqn_lr"]
        self.memory = ReplayMemory(100000)
        self.optimizer_ddqn = torch.optim.Adam(self.Qnet.parameters(),lr=self.qqdn_lr)

        self.optimizer = torch.optim.Adam(
            list(self.network.parameters()) + list(self.adapter.parameters()),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"], betas=(0.5, 0.99)
        )

        self.optimizer_disc = torch.optim.Adam(
            self.domain_classifier.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"], betas=(0.5, 0.99)
        )

        self.hparams = hparams
        self.device = device
        self.temperature = hparams["temperature"]

    def update_mcdo(self, src_x, src_y, trg_x, global_step,step, epoch, len_dataloader): # Teacher's uncertainty as rewards

        p = float(step + epoch * len_dataloader) / self.hparams["num_epochs"] + 1 / len_dataloader
        alpha = 2. / (1. + np.exp(-10 * p)) - 1

        self.network_t.eval()

        # Format Batch through teacher model
        src_feat_t = self.t_feature_extractor(src_x)
        src_feat_t = Variable(src_feat_t, requires_grad=False)

        trg_feat_t = self.t_feature_extractor(trg_x)
        trg_pred_t = self.t_classifier(trg_feat_t)
        trg_pred_t = Variable(trg_pred_t, requires_grad=False)

        for episode in range(self.episodes):
            # zero grad
            self.optimizer.zero_grad()
            self.optimizer_disc.zero_grad()

            # Student Forward with target data and get the feature maps as the state
            trg_feat_s = self.feature_extractor(trg_x)
            states_s = trg_feat_s.clone().detach()

            epsilon = max(0.01, 0.08 - 0.01 * (global_step / 50))
            weights = []

            if episode==0:
                state_list = states_s.tolist()
                action_list = []
                reward_list = []
                done_mask_list =[]
            else:
                state_next_list = states_s.tolist()
                #push to memory, to be added
                for index in range(list(states_s.size())[0]):
                    self.memory.push(state_list[index], action_list[index], np.expand_dims(reward_list[index],0),
                                     state_next_list[index],done_mask_list[index])
                state_list = state_next_list
                action_list =[]
                reward_list = []
                done_mask_list =[]

            for ii in range(list(states_s.size())[0]): # for every state in states set
                action, out, _ = self.Qnet.sample_action(states_s[ii].unsqueeze(0), epsilon)
                weights.append(action[0])

                done = False

                if episode == self.episodes-1 or action[0]==0:
                    done = True

                done_mask = 0.0 if done else 1.0
                action_list.append(action)
                done_mask_list.append(np.expand_dims(done_mask,0))

            # Calculate the rewards with MC Dropout uncertainty
            for module in self.t_feature_extractor.modules():
                if module.__class__.__name__.startswith('Dropout'):
                    module.train()
            for module in self.t_classifier.modules():
                if module.__class__.__name__.startswith('Dropout'):
                    module.train()

            forward_pass = 10
            n_samples = src_y.shape[0]
            n_classes = self.hparams["n_classes"]
            dropout_predictions = np.empty((0, n_samples, n_classes))
            softmax = nn.Softmax(dim=1)
            for i in range(forward_pass):
                predictions = np.empty((0, n_classes))
                trg_feat_temp = self.t_feature_extractor(trg_x)
                trg_pred_temp = self.t_classifier(trg_feat_temp)
                output = softmax(trg_pred_temp)
                predictions = np.vstack((predictions, output.detach().cpu().numpy()))
                dropout_predictions = np.vstack((dropout_predictions, predictions[np.newaxis, :, :]))

            mean = np.mean(dropout_predictions, axis=0)
            # variance = np.var(dropout_predictions,axis=0)
            eps = sys.float_info.min
            # Calculating entropy across multiple MCD forward passes
            entropy = -np.sum(mean * np.log(mean + eps), axis=-1)  # shape (n_samples,)

            m = np.sign(np.mean(entropy)-entropy) # 1, lower than average=high confidence, -1 higher than average=low confidence

            # Calculate reward
            for iii in range(list(states_s.size())[0]):
                if action_list[iii][0] == 1 and m[iii] > 0: # select sample with high confidence
                    reward = 1
                if action_list[iii][0] == 1 and m[iii] <= 0: # select sample with low confidence
                    reward = -1
                if action_list[iii][0] == 0 and m[iii] > 0:  # not select sample with high confidence
                    reward = -1
                if action_list[iii][0] == 0 and m[iii] <= 0:  # not select sample with low confidence
                    reward = 1
                reward_list.append(reward)

            if episode == self.episodes-1:
                # push to memory
                for index in range(list(states_s.size())[0]):
                    self.memory.push(state_next_list[index], action_list[index], np.expand_dims(reward_list[index], 0),
                                     state_next_list[index], done_mask_list[index])
            # if episode != 0:
            #     indexes = [i for i, x in enumerate(pre_weights) if x==0]
            #     for index in indexes:
            #         weights[index] = 0
            # pre_weights = weights
            # # print(weights)

            weight = torch.Tensor(weights).to(self.device)
            # Student Forward
            trg_feat_s = self.feature_extractor(trg_x)
            trg_hint_s = self.adapter(trg_feat_s)
            trg_pred_s = self.classifier(trg_feat_s)

            # Calculate the KD loss
            trg_pred_t_soften = torch.nn.functional.log_softmax(trg_pred_t / self.temperature, dim=1)
            trg_pred_s_soften = torch.nn.functional.log_softmax(trg_pred_s / self.temperature, dim=1)

            soft_loss = torch.nn.functional.kl_div(trg_pred_s_soften, trg_pred_t_soften, reduction='none',
                                                   log_target=True).sum(axis=1)
            soft_loss = (weight*soft_loss).sum()


            kd_loss = soft_loss * self.temperature ** 2

            # Calculate Domain confusion loss, reuse teacher's discriminator
            domain_label_src = torch.ones(len(src_x)).to(self.device)
            domain_label_trg = torch.zeros(len(trg_x)).to(self.device)

            # src_feat_reversed = ReverseLayerF.apply(src_feat_t, alpha)
            # src_domain_pred = self.domain_classifier(src_feat_reversed)
            src_domain_pred = self.domain_classifier(src_feat_t)
            src_domain_loss = self.cross_entropy(src_domain_pred, domain_label_src.long())

            # trg_domain_pred = ReverseLayerF.apply(trg_hint_s, alpha)
            trg_domain_pred = self.domain_classifier(trg_hint_s)
            trg_domain_loss = self.cross_entropy(trg_domain_pred, domain_label_trg.long())
            domain_loss = src_domain_loss + trg_domain_loss

            # Total loss
            loss= self.hparams["dc_loss_wt"]*domain_loss + self.hparams["kd_loss_wt"] * kd_loss
            loss.backward()
            self.optimizer.step()
            self.optimizer_disc.step()

            # Update DDQN
            if len(self.memory) > 100:
                s, a, r, s_prime, done_mask = self.memory.sample(32)

                # double dqn
                q_out = self.Qnet(torch.Tensor(s).to(self.device))
                q_a = q_out.gather(1, ((torch.Tensor(a)).type(torch.LongTensor)).to(self.device))

                a_max = self.Qnet(torch.Tensor(s_prime).to(self.device)).detach().argmax(1).unsqueeze(1)  # max(1)[1].view(-1, 1)
                max_q_prime = self.Qnet_target(torch.Tensor(s_prime).to(self.device)).gather(1, a_max)
                target = torch.Tensor(r).to(self.device) + 0.9 * max_q_prime * torch.Tensor(done_mask).to(self.device)

                loss_dqn = F.smooth_l1_loss(q_a, target)

                self.optimizer_ddqn.zero_grad()
                loss_dqn.backward()
                self.optimizer_ddqn.step()
                for target_param, param in zip(self.Qnet_target.parameters(), self.Qnet.parameters()):
                    target_param.data.copy_(target_param.data * (1.0 - 0.001) + param.data * 0.001)

                self.Qnet.reset_noise()
                self.Qnet_target.reset_noise()

        return {'Total_loss': loss.item(), 'Domain_loss': domain_loss.item(), 'KD_loss': kd_loss.item()}


    def update_mcdo_new(self, src_x, src_y, trg_x, global_step,step, epoch, len_dataloader): # student's uncertainty as rewards

        p = float(step + epoch * len_dataloader) / self.hparams["num_epochs"] + 1 / len_dataloader
        alpha = 2. / (1. + np.exp(-10 * p)) - 1

        self.network_t.eval()

        # Format Batch through teacher model
        src_feat_t = self.t_feature_extractor(src_x)
        src_feat_t = Variable(src_feat_t, requires_grad=False)

        trg_feat_t = self.t_feature_extractor(trg_x)
        trg_pred_t = self.t_classifier(trg_feat_t)
        trg_pred_t = Variable(trg_pred_t, requires_grad=False)

        for episode in range(self.episodes):
            # zero grad
            self.optimizer.zero_grad()
            self.optimizer_disc.zero_grad()

            # Student Forward with target data and get the feature maps as the state
            trg_feat_s = self.feature_extractor(trg_x)
            states_s = trg_feat_s.detach().clone()

            epsilon = max(0.01, 0.08 - 0.01 * (global_step / 50))
            weights = []

            if episode==0:
                state_list = states_s.tolist()
                action_list = []
                reward_list = []
                done_mask_list =[]
            else:
                state_next_list = states_s.tolist()
                #push to memory, to be added
                for index in range(list(states_s.size())[0]):
                    self.memory.push(state_list[index], action_list[index], np.expand_dims(reward_list[index],0),
                                     state_next_list[index],done_mask_list[index])
                state_list = state_next_list
                action_list =[]
                reward_list = []
                done_mask_list =[]

            for ii in range(list(states_s.size())[0]): # for every state in states set
                action, out, _ = self.Qnet.sample_action(states_s[ii].unsqueeze(0), epsilon)
                weights.append(action[0])

                done = False

                if episode == self.episodes-1 or action[0]==0:
                    done = True

                done_mask = 0.0 if done else 1.0
                action_list.append(action)
                done_mask_list.append(np.expand_dims(done_mask,0))

            weight = torch.Tensor(weights).to(self.device)

            # Calculate the rewards with MC Dropout uncertainty
            for module in self.t_feature_extractor.modules():
                if module.__class__.__name__.startswith('Dropout'):
                    module.train()
            for module in self.t_classifier.modules():
                if module.__class__.__name__.startswith('Dropout'):
                    module.train()

            forward_pass = 10
            n_samples = src_y.shape[0]
            n_classes = self.hparams["n_classes"]
            dropout_predictions = np.empty((0, n_samples, n_classes))
            softmax = nn.Softmax(dim=1)
            for i in range(forward_pass):
                predictions = np.empty((0, n_classes))
                trg_feat_temp = self.t_feature_extractor(trg_x)
                trg_pred_temp = self.t_classifier(trg_feat_temp)
                output = softmax(trg_pred_temp)
                predictions = np.vstack((predictions, output.detach().clone().cpu().numpy()))
                dropout_predictions = np.vstack((dropout_predictions, predictions[np.newaxis, :, :]))

            mean = np.mean(dropout_predictions, axis=0)
            # variance = np.var(dropout_predictions,axis=0)
            eps = sys.float_info.min
            # Calculating entropy across multiple MCD forward passes
            entropy = -np.sum(mean * np.log(mean + eps), axis=-1)  # shape (n_samples,)

            m = np.sign(np.mean(entropy) - entropy)  # 1, lower than average=high confidence, -1 higher than average=low confidence

            # Student Forward
            trg_feat_s = self.feature_extractor(trg_x)
            trg_hint_s = self.adapter(trg_feat_s)
            trg_pred_s = self.classifier(trg_feat_s)
            trg_hint_s_duplicate = trg_hint_s.detach().clone()  # for entoropy calculation

            # Calculate the KD loss
            trg_pred_t_soften = torch.nn.functional.log_softmax(trg_pred_t / self.temperature, dim=1)
            trg_pred_s_soften = torch.nn.functional.log_softmax(trg_pred_s / self.temperature, dim=1)

            soft_loss = torch.nn.functional.kl_div(trg_pred_s_soften, trg_pred_t_soften, reduction='none',
                                                   log_target=True).sum(axis=1)
            soft_loss = (weight * soft_loss).sum()

            kd_loss = soft_loss * self.temperature ** 2

            # Calculate Domain confusion loss, reuse teacher's discriminator
            domain_label_src = torch.ones(len(src_x)).to(self.device)
            domain_label_trg = torch.zeros(len(trg_x)).to(self.device)

            # src_feat_reversed = ReverseLayerF.apply(src_feat_t, alpha)
            # src_domain_pred = self.domain_classifier(src_feat_reversed)
            src_domain_pred = self.domain_classifier(src_feat_t)
            src_domain_loss = self.cross_entropy(src_domain_pred, domain_label_src.long())

            # trg_domain_pred = ReverseLayerF.apply(trg_hint_s, alpha)
            trg_domain_pred = self.domain_classifier(trg_hint_s)
            trg_domain_loss = self.cross_entropy(trg_domain_pred, domain_label_trg.long())
            domain_loss = src_domain_loss + trg_domain_loss

            # Total loss
            loss = self.hparams["dc_loss_wt"] * domain_loss + self.hparams["kd_loss_wt"] * kd_loss
            loss.backward()
            self.optimizer.step()
            self.optimizer_disc.step()

            #  Calculate cosine distance between teacher and student
            d = np.diagonal(cosine_distances(trg_hint_s_duplicate.cpu().numpy(), trg_feat_t.detach().clone().cpu().numpy()))
            # d = np.diagonal(euclidean_distances(trg_hint_s_duplicate.cpu().numpy(), trg_feat_t.detach().cpu().numpy()))
            n = np.sign(np.mean(d) - d) #1, easy sample, should be selected;-1, difficult sample, should not be selected

            # Calculate reward
            for iii in range(list(states_s.size())[0]):
                if action_list[iii][0] == 1 and m[iii] > 0:  # select sample with high confidence
                    reward_1 = 1
                if action_list[iii][0] == 1 and m[iii] <= 0:  # select sample with low confidence
                    reward_1 = -1
                if action_list[iii][0] == 0 and m[iii] > 0:  # not select sample with high confidence
                    reward_1 = -1
                if action_list[iii][0] == 0 and m[iii] <= 0:  # not select sample with low confidence
                    reward_1 = 1

                # if action_list[iii][0] == 1 and n[iii] > 0:  # select easy sample
                #     reward_2 = 1
                # if action_list[iii][0] == 1 and n[iii] <= 0:  # select difficult sample
                #     reward_2 = -1
                # if action_list[iii][0] == 0 and n[iii] > 0:  # not select easy sample
                #     reward_2 = -1
                # if action_list[iii][0] == 0 and n[iii] <= 0:  # not select difficult sample
                #     reward_2 = 1

                # reward = 1.8 * (reward_1-0.5) + 0.2 * (reward_2-0.5)
                reward = reward_1
                reward_list.append(reward)
            #
            if episode == self.episodes-1:
                # push to memory
                for index in range(list(states_s.size())[0]):
                    self.memory.push(state_next_list[index], action_list[index], np.expand_dims(reward_list[index], 0),
                                     state_next_list[index], done_mask_list[index])


            # Update DDQN
            if len(self.memory) > 100:
                s, a, r, s_prime, done_mask = self.memory.sample(32)

                # double dqn
                q_out = self.Qnet(torch.Tensor(s).to(self.device))
                q_a = q_out.gather(1, ((torch.Tensor(a)).type(torch.LongTensor)).to(self.device))

                a_max = self.Qnet(torch.Tensor(s_prime).to(self.device)).detach().argmax(1).unsqueeze(1)  # max(1)[1].view(-1, 1)
                max_q_prime = self.Qnet_target(torch.Tensor(s_prime).to(self.device)).gather(1, a_max)
                target = torch.Tensor(r).to(self.device) + 0.9 * max_q_prime * torch.Tensor(done_mask).to(self.device)

                loss_dqn = F.smooth_l1_loss(q_a, target)

                self.optimizer_ddqn.zero_grad()
                loss_dqn.backward()
                self.optimizer_ddqn.step()
                for target_param, param in zip(self.Qnet_target.parameters(), self.Qnet.parameters()):
                    target_param.data.copy_(target_param.data * (1.0 - 0.001) + param.data * 0.001)

                self.Qnet.reset_noise()
                self.Qnet_target.reset_noise()

        return {'Total_loss': loss.item(), 'Domain_loss': domain_loss.item(), 'KD_loss': kd_loss.item(), 'dqn_loss': loss_dqn.item()}


    def update_mcdo_new_2(self, src_x, src_y, trg_x, global_step,step, epoch, len_dataloader): # student's uncertainty as rewards

        p = float(step + epoch * len_dataloader) / self.hparams["num_epochs"] + 1 / len_dataloader
        alpha = 2. / (1. + np.exp(-10 * p)) - 1

        self.network_t.eval()
        self.network_backup.eval()

        # Format Batch through teacher model
        src_feat_t = self.t_feature_extractor(src_x)
        src_feat_t = Variable(src_feat_t, requires_grad=False)

        trg_feat_t = self.t_feature_extractor(trg_x)
        trg_pred_t = self.t_classifier(trg_feat_t)
        trg_pred_t = Variable(trg_pred_t, requires_grad=False)

        for episode in range(self.episodes):
            # zero grad
            self.optimizer.zero_grad()
            self.optimizer_disc.zero_grad()

            # Student Forward with target data and get the feature maps as the state
            trg_feat_s = self.feature_extractor(trg_x)
            states_s = trg_feat_s.detach().clone()

            epsilon = max(0.01, 0.08 - 0.01 * (global_step / 50))
            weights = []

            if episode==0:
                state_list = states_s.tolist()
                action_list = []
                reward_list = []
                done_mask_list =[]
            else:
                state_next_list = states_s.tolist()
                #push to memory, to be added
                for index in range(list(states_s.size())[0]):
                    self.memory.push(state_list[index], action_list[index], np.expand_dims(reward_list[index],0),
                                     state_next_list[index],done_mask_list[index])
                state_list = state_next_list
                action_list =[]
                reward_list = []
                done_mask_list =[]

            for ii in range(list(states_s.size())[0]): # for every state in states set
                action, out, _ = self.Qnet.sample_action(states_s[ii].unsqueeze(0), epsilon)
                weights.append(action[0])
                done = False
                if episode == self.episodes-1 or action[0]==0:
                    done = True
                done_mask = 0.0 if done else 1.0
                action_list.append(action)
                done_mask_list.append(np.expand_dims(done_mask,0))

            # if episode != 0:
            #     indexes = [i for i, x in enumerate(pre_weights) if x==0]
            #     for index in indexes:
            #         weights[index] = 0
            # pre_weights = weights
            # print(weights)
            weight = torch.Tensor(weights).to(self.device)

            # Calculate the rewards with MC Dropout uncertainty
            for module in self.t_feature_extractor.modules():
                if module.__class__.__name__.startswith('Dropout'):
                    module.train()
            for module in self.t_classifier.modules():
                if module.__class__.__name__.startswith('Dropout'):
                    module.train()

            forward_pass = 10
            n_samples = src_y.shape[0]
            n_classes = self.hparams["n_classes"]
            dropout_predictions_t = np.empty((0, n_samples, n_classes))
            softmax = nn.Softmax(dim=1)
            for i in range(forward_pass):
                predictions = np.empty((0, n_classes))
                trg_feat_temp = self.t_feature_extractor(trg_x)
                trg_pred_temp = self.t_classifier(trg_feat_temp)
                output = softmax(trg_pred_temp)
                predictions = np.vstack((predictions, output.detach().clone().cpu().numpy()))
                dropout_predictions_t = np.vstack((dropout_predictions_t, predictions[np.newaxis, :, :]))

            mean_t = np.mean(dropout_predictions_t, axis=0)
            # variance = np.var(dropout_predictions,axis=0)
            eps = sys.float_info.min
            # Calculating entropy across multiple MCD forward passes
            entropy_t = -np.sum(mean_t * np.log(mean_t + eps), axis=-1)  # shape (n_samples,)
            m_t = np.sign(np.mean(entropy_t) - entropy_t)  # 1, lower than average=high confidence, -1 higher than average=low confidence

            # Student Forward
            trg_feat_s = self.feature_extractor(trg_x)
            trg_hint_s = self.adapter(trg_feat_s)
            trg_pred_s = self.classifier(trg_feat_s)
            trg_pred_s_duplicate = trg_pred_s.detach().clone()  # for entoropy calculation

            # Calculate the KD loss
            trg_pred_t_soften = torch.nn.functional.log_softmax(trg_pred_t / self.temperature, dim=1)
            trg_pred_s_soften = torch.nn.functional.log_softmax(trg_pred_s / self.temperature, dim=1)

            soft_loss = torch.nn.functional.kl_div(trg_pred_s_soften, trg_pred_t_soften, reduction='none',
                                                   log_target=True).sum(axis=1)
            soft_loss = (weight * soft_loss).sum()

            kd_loss = soft_loss * self.temperature ** 2

            # Calculate Domain confusion loss, reuse teacher's discriminator
            domain_label_src = torch.ones(len(src_x)).to(self.device)
            domain_label_trg = torch.zeros(len(trg_x)).to(self.device)

            # src_feat_reversed = ReverseLayerF.apply(src_feat_t, alpha)
            # src_domain_pred = self.domain_classifier(src_feat_reversed)
            src_domain_pred = self.domain_classifier(src_feat_t)
            src_domain_loss = self.cross_entropy(src_domain_pred, domain_label_src.long())

            # trg_domain_pred = ReverseLayerF.apply(trg_hint_s, alpha)
            trg_domain_pred = self.domain_classifier(trg_hint_s)
            trg_domain_loss = self.cross_entropy(trg_domain_pred, domain_label_trg.long())
            domain_loss = src_domain_loss + trg_domain_loss

            # Total loss
            loss = self.hparams["dc_loss_wt"] * domain_loss + self.hparams["kd_loss_wt"] * kd_loss
            loss.backward()
            self.optimizer.step()
            self.optimizer_disc.step()

            # self.feature_extractor_backup.load_state_dict(self.feature_extractor.state_dict())
            # self.classifier_backup.load_state_dict(self.classifier.state_dict())
            # # Calculate the rewards with MC Dropout uncertainty
            # for module in self.network_backup.modules():
            #     if module.__class__.__name__.startswith('Dropout'):
            #         module.train()
            #
            # dropout_predictions_s = np.empty((0, n_samples, n_classes))
            # for i in range(forward_pass):
            #     predictions = np.empty((0, n_classes))
            #     trg_feat_temp = self.feature_extractor_backup(trg_x)
            #     trg_pred_temp = self.classifier_backup(trg_feat_temp)
            #     output = softmax(trg_pred_temp)
            #     predictions = np.vstack((predictions, output.detach().clone().cpu().numpy()))
            #     dropout_predictions_s = np.vstack((dropout_predictions_s, predictions[np.newaxis, :, :]))
            #
            # mean_s = np.mean(dropout_predictions_s, axis=0)
            # entropy_s = -np.sum(mean_s * np.log(mean_s + eps), axis=-1)  # shape (n_samples,)
            # m_s = np.sign(np.mean(entropy_s) - entropy_s)


            trg_pred_s_duplicate = softmax(trg_pred_s_duplicate).cpu().numpy()
            entropy_s = -np.sum(trg_pred_s_duplicate * np.log(trg_pred_s_duplicate + eps), axis=-1)  # shape (n_samples,)
            m_s = np.sign(np.mean(entropy_s) - entropy_s)

            #  Calculate kl distance between teacher and student
            d = np.sum(kl_div(trg_pred_s_duplicate, mean_t),axis=-1)
            n = np.sign(np.mean(d)-d)

            # Calculate reward
            for iii in range(list(states_s.size())[0]):
                # if action_list[iii][0] == 1 and (m_s[iii] == m_t[iii]):  # select sample with high confidence
                #     reward_1 = 1
                # if action_list[iii][0] == 1 and (m_s[iii] != m_t[iii]):  # select sample with low confidence
                #     reward_1 = -1
                # if action_list[iii][0] == 0 and (m_s[iii] == m_t[iii]):  # not select sample with high confidence
                #     reward_1 = -1
                # if action_list[iii][0] == 0 and (m_s[iii] != m_t[iii]):  # not select sample with low confidence
                #     reward_1 = 1

                if action_list[iii][0] == 1 and m_s[iii] == 1 and m_t[iii]==1:  # select sample with high confidence
                    reward_1 = 1
                elif action_list[iii][0] == 0 and m_s[iii] == -1 and m_t[iii]==-1:
                    reward_1 = 1
                else:
                    reward_1 =-1


                if action_list[iii][0] == 1 and n[iii] > 0:  # select easy sample
                    reward_2 = 1
                if action_list[iii][0] == 1 and n[iii] <= 0:  # select difficult sample
                    reward_2 = -1
                if action_list[iii][0] == 0 and n[iii] > 0:  # not select easy sample
                    reward_2 = -1
                if action_list[iii][0] == 0 and n[iii] <= 0:  # not select difficult sample
                    reward_2 = 1

                reward = 0.2 * (reward_1-0.5) + 1.8 * (reward_2-0.5)
                # reward = reward_2
                reward_list.append(reward)
            #
            if episode == self.episodes-1:
                # push to memory
                for index in range(list(states_s.size())[0]):
                    self.memory.push(state_next_list[index], action_list[index], np.expand_dims(reward_list[index], 0),
                                     state_next_list[index], done_mask_list[index])


            # Update DDQN
            if len(self.memory) > 100:
                s, a, r, s_prime, done_mask = self.memory.sample(32)

                # double dqn
                q_out = self.Qnet(torch.Tensor(s).to(self.device))
                q_a = q_out.gather(1, ((torch.Tensor(a)).type(torch.LongTensor)).to(self.device))

                a_max = self.Qnet(torch.Tensor(s_prime).to(self.device)).detach().argmax(1).unsqueeze(1)  # max(1)[1].view(-1, 1)
                max_q_prime = self.Qnet_target(torch.Tensor(s_prime).to(self.device)).gather(1, a_max)
                target = torch.Tensor(r).to(self.device) + 0.9 * max_q_prime * torch.Tensor(done_mask).to(self.device)

                loss_dqn = F.smooth_l1_loss(q_a, target)

                self.optimizer_ddqn.zero_grad()
                loss_dqn.backward()
                self.optimizer_ddqn.step()
                for target_param, param in zip(self.Qnet_target.parameters(), self.Qnet.parameters()):
                    target_param.data.copy_(target_param.data * (1.0 - 0.001) + param.data * 0.001)

                self.Qnet.reset_noise()
                self.Qnet_target.reset_noise()

        return {'Total_loss': loss.item(), 'Domain_loss': domain_loss.item(), 'KD_loss': kd_loss.item(), 'dqn_loss': loss_dqn.item()}


class DAKD(Algorithm):
    """
    DAKD
    """

    def __init__(self, backbone_fe, configs, hparams, device):
        super(DAKD, self).__init__(configs)
        from models import models
        # Teacher Model and discriminator
        self.t_src_encoder= models.CNN_T(configs)
        self.t_tgt_encoder = models.CNN_T(configs)
        self.t_src_classifier = models.classifier_T(configs)
        self.network_t = nn.Sequential(self.t_src_encoder, self.t_src_classifier)

        # self.domain_classifier = Discriminator_t(configs)
        self.domain_classifier = Discriminator_fea(configs)

        self.feature_extractor = backbone_fe(configs)
        self.classifier = classifier(configs)
        self.network_s = nn.Sequential(self.feature_extractor, self.classifier)
        self.criterion = nn.CrossEntropyLoss()
        # self.criterion = nn.BCELoss()

        # optimizer for teacher training on source
        self.optimizer_t_src = torch.optim.Adam(
            list(self.t_src_encoder.parameters()) + list(self.t_src_classifier.parameters()),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"], betas=(0.5, 0.99)
        )

        # optimizer for teacher training on target
        self.optimizer_t_tgt = torch.optim.Adam(
            self.t_tgt_encoder.parameters(),
            lr=hparams["learning_rate"]*0.1,
            weight_decay=hparams["weight_decay"], betas=(0.5, 0.99)
        )


        # optimizer for discriminator
        self.optimizer_critic = torch.optim.Adam(
            self.domain_classifier.parameters(),
            lr=hparams["dis_learning_rate"],
            weight_decay=hparams["weight_decay"], betas=(0.5, 0.99)
        )


        #optimizer for student's training
        self.optimizer = torch.optim.Adam(
            self.network_s.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"], betas=(0.5, 0.99)
        )

        self.hparams = hparams
        self.device = device
        self.temperature = hparams["temperature"]

    def update_src(self, src_x, src_y):

        self.optimizer_t_src.zero_grad()
        src_feat = self.t_src_encoder(src_x)
        src_pred = self.t_src_classifier(src_feat)
        src_cls_loss = self.cross_entropy(src_pred, src_y)
        loss = src_cls_loss
        loss.backward()
        self.optimizer_t_src.step()
        return {'src_cls_loss': src_cls_loss.item()}

    def update_tgt(self, src_x, trg_x):

        ###########################
        # 2.1 train discriminator #
        ###########################
        # zero gradients for optimizer
        self.optimizer_critic.zero_grad()

        # extract and concat features
        feat_src = self.t_src_encoder(src_x)
        feat_tgt = self.t_tgt_encoder(trg_x)
        feat_concat = torch.cat((feat_src, feat_tgt), 0)
        # feat_concat = feat_concat.view(feat_concat.size(0), -1)

        # predict on discriminator
        # pred_concat = self.domain_classifier(feat_concat.detach()).view(-1)
        pred_concat = self.domain_classifier(feat_concat.detach())

        # prepare real and fake label
        label_src = Variable(torch.zeros(feat_src.size(0)).long().cuda())
        label_tgt = Variable(torch.ones(feat_tgt.size(0)).long().cuda())
        label_concat = torch.cat((label_src, label_tgt), 0)
        # compute loss for critic
        loss_critic = self.criterion(pred_concat, label_concat)
        loss_critic.backward()

        # optimize critic
        self.optimizer_critic.step()

        ############################
        # 2.2 train target encoder #
        ############################
        # zero gradients for optimizer
        self.optimizer_critic.zero_grad()
        self.optimizer_t_tgt.zero_grad()

        # extract target features
        feat_tgt = self.t_tgt_encoder(trg_x)
        # feat_tgt = feat_tgt.view(feat_tgt.size(0), -1)

        # predict on discriminator
        # pred_tgt = self.domain_classifier(feat_tgt).view(-1)
        pred_tgt = self.domain_classifier(feat_tgt)

        # prepare fake labels
        label_tgt = Variable(torch.zeros(feat_tgt.size(0)).long().cuda())

        # compute loss for target encoder
        loss = self.criterion(pred_tgt, label_tgt)
        loss.backward()

        self.optimizer_t_tgt.step()

        return {'discriminator_loss': loss_critic.item(),'t_tgt_encoder_loss': loss.item()}

    def update_kd(self, trg_x):

        ####################
        # 1. setup network #
        ####################

        # set train state for Dropout and BN layers
        self.t_tgt_encoder.eval()
        self.t_src_classifier.eval()
        self.network_s.train()

        # zero gradients for optimizer
        self.optimizer.zero_grad()

        # Forward through teacher
        feat_tgt = self.t_tgt_encoder(trg_x)
        trg_pred_t = self.t_src_classifier(feat_tgt)

        # Forward through student
        # feat_tgt_s = self.feature_extractor(trg_x)
        # trg_pred_s = self.classifier(feat_tgt_s)
        # hint_s = self.adapter(feat_tgt_s)
        trg_pred_s = self.network_s(trg_x)

        # Calculate the KD loss
        trg_pred_t_soften = torch.nn.functional.log_softmax(trg_pred_t / self.temperature, dim=1)
        trg_pred_s_soften = torch.nn.functional.log_softmax(trg_pred_s / self.temperature, dim=1)
        loss = torch.nn.functional.kl_div(trg_pred_s_soften, trg_pred_t_soften, reduction='batchmean', log_target=True)

        # loss_hint = torch.nn.functional.l1_loss(hint_s, feat_tgt)
        # loss = loss_kd + loss_hint
        loss.backward()

        # optimize
        self.optimizer.step()
        # return {'kd_loss': loss_kd.item(),'hint_loss': loss_hint.item()}
        return {'kd_loss': loss.item()}


    def update(self, src_x, src_y, trg_x, step, epoch, len_dataloader):

        self.network_t.eval()
        real_label = 1
        fake_label = 0

        # zero grad
        self.optimizer.zero_grad()
        # Format Batch through teacher model
        src_feat_t = self.t_feature_extractor(src_x)
        src_feat_t = Variable(src_feat_t, requires_grad=False)

        trg_feat_t = self.t_feature_extractor(trg_x)
        trg_pred_t = self.t_classifier(trg_feat_t)
        trg_pred_t = Variable(trg_pred_t, requires_grad=False)

        # Student Forward
        trg_feat_s = self.feature_extractor(trg_x)
        trg_hint_s = self.adapter(trg_feat_s)
        trg_pred_s = self.classifier(trg_feat_s)

        # Calculate the KD loss
        trg_pred_t_soften = torch.nn.functional.log_softmax(trg_pred_t / self.temperature, dim=1)
        trg_pred_s_soften = torch.nn.functional.log_softmax(trg_pred_s / self.temperature, dim=1)
        soft_loss = torch.nn.functional.kl_div(trg_pred_s_soften, trg_pred_t_soften, reduction='batchmean', log_target=True)
        kd_loss = soft_loss * self.temperature ** 2

        # Calculate Domain confusion loss, reuse teacher's discriminator
        domain_label_src = torch.ones(len(src_x)).to(self.device)
        domain_label_trg = torch.zeros(len(trg_x)).to(self.device)

        src_domain_pred = self.domain_classifier(src_feat_t)
        src_domain_loss = self.cross_entropy(src_domain_pred, domain_label_src.long())
        trg_domain_pred = self.domain_classifier(trg_hint_s)
        trg_domain_loss = self.cross_entropy(trg_domain_pred, domain_label_trg.long())
        domain_loss = src_domain_loss + trg_domain_loss

        # Total loss
        loss= domain_loss +  self.hparams["kd_loss_wt"] * kd_loss
        loss.backward()
        self.optimizer.step()

        return {'Total_loss': loss.item(), 'Domain_loss': domain_loss.item(), 'KD_loss': kd_loss.item()}


class KDDA(Algorithm):
    """
    KDDA
    """

    def __init__(self, backbone_fe, configs, hparams, device):
        super(KDDA, self).__init__(configs)
        from models import models
        # Teacher Model and discriminator
        self.t_src_encoder= models.CNN_T(configs)
        self.t_src_classifier = models.classifier_T(configs)
        self.network_t = nn.Sequential(self.t_src_encoder, self.t_src_classifier)

        # self.domain_classifier = Discriminator_t(configs)
        self.domain_classifier = Discriminator_s(configs)

        self.s_src_encoder = backbone_fe(configs)
        self.s_src_classifier = classifier(configs)
        self.s_tgt_encoder = backbone_fe(configs)

        self.criterion = nn.CrossEntropyLoss()
        # self.criterion = nn.BCELoss()

        # optimizer for teacher training on source
        self.optimizer_t_src = torch.optim.Adam(
            list(self.t_src_encoder.parameters()) + list(self.t_src_classifier.parameters()),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"], betas=(0.5, 0.99)
        )

        # optimizer for teacher training on target
        self.optimizer_s_tgt_encoder = torch.optim.Adam(
            self.s_tgt_encoder.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"], betas=(0.5, 0.99)
        )


        # optimizer for discriminator
        self.optimizer_critic = torch.optim.Adam(
            self.domain_classifier.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"], betas=(0.5, 0.99)
        )

        # optimizer for student's training
        self.optimizer_kd = torch.optim.Adam(
            list(self.s_src_encoder.parameters())+list(self.s_src_classifier.parameters()),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"], betas=(0.5, 0.99)
        )
        self.hparams = hparams
        self.device = device
        self.temperature = hparams["temperature"]

    def update_src(self, src_x, src_y):

        self.optimizer_t_src.zero_grad()
        src_feat = self.t_src_encoder(src_x)
        src_pred = self.t_src_classifier(src_feat)
        src_cls_loss = self.cross_entropy(src_pred, src_y)
        loss = src_cls_loss
        loss.backward()
        self.optimizer_t_src.step()
        return {'src_cls_loss': src_cls_loss.item()}

    def update_tgt(self, src_x, trg_x):

        ###########################
        # 2.1 train discriminator #
        ###########################
        # zero gradients for optimizer
        self.optimizer_critic.zero_grad()

        # extract and concat features
        feat_src = self.s_src_encoder(src_x)
        feat_tgt = self.s_tgt_encoder(trg_x)
        feat_concat = torch.cat((feat_src, feat_tgt), 0)
        # feat_concat = feat_concat.view(feat_concat.size(0), -1)

        # predict on discriminator
        # pred_concat = self.domain_classifier(feat_concat.detach()).view(-1)
        pred_concat = self.domain_classifier(feat_concat.detach())

        # prepare real and fake label
        label_src = Variable(torch.zeros(feat_src.size(0)).long().cuda())
        label_tgt = Variable(torch.ones(feat_tgt.size(0)).long().cuda())
        label_concat = torch.cat((label_src, label_tgt), 0)
        # compute loss for critic
        loss_critic = self.criterion(pred_concat, label_concat)
        loss_critic.backward()

        # optimize critic
        self.optimizer_critic.step()

        ############################
        # 2.2 train target encoder #
        ############################
        # zero gradients for optimizer
        self.optimizer_critic.zero_grad()
        self.optimizer_s_tgt_encoder.zero_grad()

        # extract target features
        feat_tgt = self.s_tgt_encoder(trg_x)
        # feat_tgt = feat_tgt.view(feat_tgt.size(0), -1)

        # predict on discriminator
        # pred_tgt = self.domain_classifier(feat_tgt).view(-1)
        pred_tgt = self.domain_classifier(feat_tgt)

        # prepare fake labels
        label_tgt = Variable(torch.zeros(feat_tgt.size(0)).long().cuda())

        # compute loss for target encoder
        loss = self.criterion(pred_tgt, label_tgt)
        loss.backward()

        self.optimizer_s_tgt_encoder.step()

        return {'discriminator_loss': loss_critic.item(),'t_tgt_encoder_loss': loss.item()}

    def update_kd(self, src_x, src_y):

        ####################
        # 1. setup network #
        ####################

        # zero gradients for optimizer
        self.optimizer_kd.zero_grad()

        # Forward through teacher
        feat_src = self.t_src_encoder(src_x)
        src_pred_t = self.t_src_classifier(feat_src)

        # Forward through student
        feat_src = self.s_src_encoder(src_x)
        src_pred_s = self.s_src_classifier(feat_src)

        # Calculate the KD loss
        src_pred_t_soften = torch.nn.functional.log_softmax(src_pred_t / self.temperature, dim=1)
        src_pred_s_soften = torch.nn.functional.log_softmax(src_pred_s / self.temperature, dim=1)
        loss_kd = torch.nn.functional.kl_div(src_pred_s_soften, src_pred_t_soften, reduction='batchmean', log_target=True)

        # Calculate CE loss
        loss_ce = self.cross_entropy(src_pred_s, src_y)

        loss = loss_ce + self.hparams["kd_loss_wt"] * loss_kd * self.temperature ** 2

        loss.backward()

        # optimize
        self.optimizer_kd.step()

        return {'kd_loss': loss.item()}

    def update(self, src_x, src_y, trg_x, step, epoch, len_dataloader):

        self.network_t.eval()
        real_label = 1
        fake_label = 0

        is_mc_dropout =True

        # zero grad
        self.optimizer.zero_grad()
        # Format Batch through teacher model
        src_feat_t = self.t_feature_extractor(src_x)
        src_feat_t = Variable(src_feat_t, requires_grad=False)


        trg_feat_t = self.t_feature_extractor(trg_x)
        trg_pred_t = self.t_classifier(trg_feat_t)
        trg_pred_t = Variable(trg_pred_t, requires_grad=False)

        # Student Forward
        trg_feat_s = self.feature_extractor(trg_x)
        trg_hint_s = self.adapter(trg_feat_s)
        trg_pred_s = self.classifier(trg_feat_s)

        # Calculate the KD loss
        trg_pred_t_soften = torch.nn.functional.log_softmax(trg_pred_t / self.temperature, dim=1)
        trg_pred_s_soften = torch.nn.functional.log_softmax(trg_pred_s / self.temperature, dim=1)
        soft_loss = torch.nn.functional.kl_div(trg_pred_s_soften, trg_pred_t_soften, reduction='batchmean', log_target=True)
        kd_loss = soft_loss * self.temperature ** 2

        # Calculate Domain confusion loss, reuse teacher's discriminator
        domain_label_src = torch.ones(len(src_x)).to(self.device)
        domain_label_trg = torch.zeros(len(trg_x)).to(self.device)

        src_domain_pred = self.domain_classifier(src_feat_t)
        src_domain_loss = self.cross_entropy(src_domain_pred, domain_label_src.long())
        trg_domain_pred = self.domain_classifier(trg_hint_s)
        trg_domain_loss = self.cross_entropy(trg_domain_pred, domain_label_trg.long())
        domain_loss = src_domain_loss + trg_domain_loss

        # Total loss
        loss= domain_loss +  self.hparams["kd_loss_wt"] * kd_loss
        loss.backward()
        self.optimizer.step()

        return {'Total_loss': loss.item(), 'Domain_loss': domain_loss.item(), 'KD_loss': kd_loss.item()}


class MCD(Algorithm):
    """
    MCD: maximum cluster Difference
    Paper: Knowledge Adaptation: Teaching to Adapt
    """

    def __init__(self, backbone_fe, configs, hparams, device):
        super(MCD, self).__init__(configs)
        from models import models
        self.t_feature_extractor = models.CNN_T(configs)
        self.t_classifier = models.classifier_T(configs)
        self.network_t = nn.Sequential(self.t_feature_extractor, self.t_classifier)

        self.feature_extractor = backbone_fe(configs)
        self.classifier = classifier(configs)
        self.network = nn.Sequential(self.feature_extractor, self.classifier)

        self.optimizer_t = torch.optim.Adam(
            self.network_t.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"], betas=(0.5, 0.99)
        )

        self.optimizer_s = torch.optim.Adam(
            self.network.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"], betas=(0.5, 0.99)
        )

        self.hparams = hparams
        self.device = device

    def update_t(self, src_x, src_y):

        output_t = self.network_t(src_x)
        src_cls_loss = self.cross_entropy(output_t, src_y)

        loss = src_cls_loss

        self.optimizer_t.zero_grad()
        loss.backward()
        self.optimizer_t.step()
        return {'teacher src_cls_loss': src_cls_loss.item()}

    def update_s(self, trg_x, clusters):
        output_s = self.network(trg_x)
        output_s=torch.nn.functional.softmax(output_s)

        output_t = self.network_t(trg_x)
        output_t = torch.nn.functional.softmax(output_t)

        max_idx = torch.argmax(output_t, 0, keepdim=True)
        one_hot = torch.FloatTensor(output_t.shape).to(self.device)
        one_hot.zero_()
        one_hot.scatter_(0, max_idx, 1)
        beta = 0.2
        pseudo_label = (1-beta) * one_hot + beta * output_t
        loss = self.cross_entropy(output_s, pseudo_label)

        self.optimizer_s.zero_grad()
        loss.backward()
        self.optimizer_s.step()
        return {'student mcd_loss': loss.item()}

    def mcd_loss(self, predictions, clusters):
        loss = 0
        num_class = clusters.shape[1]
        for i in range(num_class-1):
            for j in range(i+1, num_class):
                 loss +=torch.abs(torch.nn.functional.cosine_similarity(predictions, clusters[i])-
                                  torch.nn.functional.cosine_similarity(predictions, clusters[j]))
        loss = loss.sum() /(num_class*(num_class-1)/2)
        return loss


class MLD(Algorithm):
    """
    Multi-level Distillation
    Paper: Domain Adaptive Knowledge Distillation for Driving Scene Semantic Segmentation
    """

    def __init__(self, backbone_fe, configs, hparams, device):
        super(MLD, self).__init__(configs)

        from models import models
        self.t_feature_extractor = models.CNN_T(configs)
        self.t_classifier = models.classifier_T(configs)
        self.network_t = nn.Sequential(self.t_feature_extractor, self.t_classifier)
        self.adapter = Adapter(configs)

        self.feature_extractor = backbone_fe(configs)
        self.classifier = classifier(configs)
        self.network = nn.Sequential(self.feature_extractor, self.classifier)

        self.optimizer = torch.optim.Adam(
            list(self.network.parameters()) + list(self.adapter.parameters()),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"], betas=(0.5, 0.99)
        )

        self.hparams = hparams

    def update_s_both(self, src_x, src_y, trg_x):

        # zero grad
        self.optimizer.zero_grad()

        #student inference
        src_feat_s = self.feature_extractor(src_x)
        src_hint_s = self.adapter(src_feat_s)
        src_pred_s = self.classifier(src_feat_s)
        trg_feat_s = self.feature_extractor(trg_x)
        trg_hint_s = self.adapter(trg_feat_s)
        trg_pred_s = self.classifier(trg_feat_s)

        # teacher inference
        src_feat_t = self.t_feature_extractor(src_x)
        src_pred_t = self.t_classifier(src_feat_t)
        trg_feat_t = self.t_feature_extractor(trg_x)
        trg_pred_t = self.t_classifier(trg_feat_t)

        # Calculate the KD loss on src
        src_pred_t= torch.nn.functional.log_softmax(src_pred_t, dim=1)
        src_pred_s= torch.nn.functional.log_softmax(src_pred_s, dim=1)
        src_kl_loss = torch.nn.functional.kl_div(src_pred_s, src_pred_t,reduction='batchmean',log_target=True)
        src_mse_loss = torch.nn.functional.mse_loss(src_feat_t, src_hint_s)
        src_pseudo_loss = self.cross_entropy(src_pred_s, torch.nn.functional.softmax(src_pred_t))

        trg_pred_t= torch.nn.functional.log_softmax(trg_pred_t, dim=1)
        trg_pred_s= torch.nn.functional.log_softmax(trg_pred_s, dim=1)
        trg_kl_loss = torch.nn.functional.kl_div(trg_pred_s, trg_pred_t,reduction='batchmean',log_target=True)
        trg_mse_loss = torch.nn.functional.mse_loss(trg_feat_t, trg_hint_s)
        trg_pseudo_loss = self.cross_entropy(trg_pred_s, torch.nn.functional.softmax(trg_pred_t))

        src_domain_loss = src_kl_loss + src_mse_loss + src_pseudo_loss
        tgt_domain_loss = trg_kl_loss + trg_mse_loss + trg_pseudo_loss

        loss = src_domain_loss + self.hparams["tgt_loss_wt"] * tgt_domain_loss
        loss.backward()
        self.optimizer.step()

        return {'Total_Student': loss.item(), 'src_domain_loss': src_domain_loss.item(), 'tgt_domain_loss': tgt_domain_loss.item()}


    def update_s_tgt(self, src_x, src_y, trg_x):

        # zero grad
        self.optimizer.zero_grad()

        #student inference

        trg_feat_s = self.feature_extractor(trg_x)
        trg_hint_s = self.adapter(trg_feat_s)
        trg_pred_s = self.classifier(trg_feat_s)

        # teacher inference

        trg_feat_t = self.t_feature_extractor(trg_x)
        trg_pred_t = self.t_classifier(trg_feat_t)

        # Calculate the KD loss on src

        trg_pred_t= torch.nn.functional.log_softmax(trg_pred_t, dim=1)
        trg_pred_s= torch.nn.functional.log_softmax(trg_pred_s, dim=1)
        trg_kl_loss = torch.nn.functional.kl_div(trg_pred_s, trg_pred_t,reduction='batchmean',log_target=True)
        trg_mse_loss = torch.nn.functional.mse_loss(trg_feat_t, trg_hint_s)
        trg_pseudo_loss = self.cross_entropy(trg_pred_s, torch.nn.functional.softmax(trg_pred_t))

        tgt_domain_loss = trg_kl_loss + trg_mse_loss + trg_pseudo_loss

        loss = tgt_domain_loss
        loss.backward()
        self.optimizer.step()

        return {'Total_Student': loss.item(), 'tgt_domain_loss': tgt_domain_loss.item()}


class REDA(Algorithm):
    """
    REDA
    PAPER: Resource Efficient Domain Adaptation
    """

    def __init__(self, backbone_fe, configs, hparams, device):
        super(REDA, self).__init__(configs)
        from models import models
        # Teacher Model and discriminator
        self.network = models.CNN_mul_exit(configs)

        self.domain_classifier = Discriminator(configs)

        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"], betas=(0.5, 0.99)
        )
        self.optimizer_disc = torch.optim.Adam(
            self.domain_classifier.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"], betas=(0.5, 0.99)
        )
        self.hparams = hparams
        self.device = device
        self.temperature = hparams["temperature"]

    def update(self, src_x, src_y, trg_x, step, epoch, len_dataloader):
        p = float(step + epoch * len_dataloader) / self.hparams["num_epochs"] + 1 / len_dataloader
        alpha = 2. / (1. + np.exp(-10 * p)) - 1

        # zero grad
        self.optimizer.zero_grad()
        self.optimizer_disc.zero_grad()

        domain_label_src = torch.ones(len(src_x)).to(self.device)
        domain_label_trg = torch.zeros(len(trg_x)).to(self.device)

        src_l, src_m, src_t, src_feat = self.network(src_x)

        trg_l, trg_m, trg_t, trg_feat = self.network(trg_x)

        # Task classification  Loss
        src_cls_loss = self.cross_entropy(src_l.squeeze(), src_y) + self.cross_entropy(src_m.squeeze(), src_y) + self.cross_entropy(src_t.squeeze(), src_y)

        # Domain classification loss on top classifier
        # source
        src_feat_reversed = ReverseLayerF.apply(src_feat, alpha)
        src_domain_pred = self.domain_classifier(src_feat_reversed)
        src_domain_loss = self.cross_entropy(src_domain_pred, domain_label_src.long())

        # target
        trg_feat_reversed = ReverseLayerF.apply(trg_feat, alpha)
        trg_domain_pred = self.domain_classifier(trg_feat_reversed)
        trg_domain_loss = self.cross_entropy(trg_domain_pred, domain_label_trg.long())

        # Total domain loss
        domain_loss = src_domain_loss + trg_domain_loss

        # Transdistill loss
        trg_t_soften = torch.nn.functional.softmax(trg_t / self.temperature, dim=1)
        trg_l_soften = torch.nn.functional.softmax(trg_l / self.temperature, dim=1)
        trg_m_soften = torch.nn.functional.softmax(trg_m / self.temperature, dim=1)

        # Calculate consistency weight
        trg_x_1 = jitter(trg_x,device=self.device)
        trg_x_2 = jitter(trg_x,device=self.device)
        _, _, trg_t_x1, _ = self.network(trg_x_1)
        _, _, trg_t_x2, _ = self.network(trg_x_2)

        trg_t_y1 = torch.argmax(trg_t_x1, 1, keepdim=True)
        trg_t_y2 = torch.argmax(trg_t_x2, 1, keepdim=True)

        alpha = 0.5 # based on paper
        consist_weight = torch.sum(trg_t_y1==trg_t_y2) + alpha * torch.sum(trg_t_y1!=trg_t_y2)

        td_loss = JSD()(trg_t_soften, trg_m_soften) + JSD()(trg_t_soften,trg_l_soften)

        loss = domain_loss + src_cls_loss + consist_weight * td_loss

        loss.backward()
        self.optimizer.step()
        self.optimizer_disc.step()

        return {'Total_loss': loss.item(), 'Domain_loss': domain_loss.item(), 'Src_cls_loss': src_cls_loss.item(), 'td_loss':td_loss.item() }


class JSD(nn.Module):
    def __init__(self):
        super(JSD, self).__init__()
        self.kl = nn.KLDivLoss(reduction='batchmean', log_target=True)

    def forward(self, p: torch.tensor, q: torch.tensor):
        p, q = p.view(-1, p.size(-1)), q.view(-1, q.size(-1))
        m = (0.5 * (p + q)).log()
        return 0.5 * (self.kl(m, p.log()) + self.kl(m, q.log()))
