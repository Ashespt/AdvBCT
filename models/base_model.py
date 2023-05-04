import timm
import torch.nn as nn
import torch
import os.path as osp
from loss.reid.metric_learning import Arcface, Cosface, AMSoftmax, CircleLoss
from models.modules import MLP,ReverseLayerF,Transformation,ElasticBoundary
class CreateBaseModel(nn.Module):
    def __init__(self,configs):
        '''
        :param args: {"arch":,
                     "pretrained": true or false,
                     "pretrained_path":str or None,
                     "num_class":,
                     "use_cls":true or false,
                     "out_dim:,
                     "adversarial":False}
        '''
        print(configs)
        super().__init__()
        if 'adversarial' not in [*configs]:
            configs['adversarial'] = False
        if 'eboundary' not in [*configs]:
            configs['eboundary'] = False
        if configs['arch'] in timm.list_models(pretrained=True):
            self.convnet = timm.create_model(configs['arch'], pretrained=configs['pretrained'], num_classes=0)
        else:
            raise Exception("not supported model architecture")

        self.encoder_dim = self.convnet.num_features
        self.use_cls = configs['use_cls']
        self.out_dim = configs['out_dim']
        self.projection_dim = nn.Linear(self.encoder_dim, configs['out_dim'], bias=False)
        self.bn = nn.BatchNorm1d(configs['out_dim'])
        self.configs = configs
        if configs['use_cls']:
            self.projection_cls = nn.Linear(configs['out_dim'], configs['num_class'], bias=False)


        if configs['adversarial']:
            self.discriminator = nn.Sequential()
            self.discriminator.add_module('d_fc1', nn.Linear(configs['out_dim'], 100))
            self.discriminator.add_module('d_bn1', nn.BatchNorm1d(100))
            self.discriminator.add_module('d_relu1', nn.ReLU(True))
            self.discriminator.add_module('d_fc2', nn.Linear(100, 2))
            self.discriminator.add_module('d_softmax', nn.LogSoftmax(dim=1))

        if configs['eboundary']:
            self.eboundary = ElasticBoundary(configs['num_class'])

        if configs['pretrained_path'] is not None and osp.exists(configs['pretrained_path']):
            self.load_state(configs['pretrained_path'])

    def load_state(self, path):
        ckpt = torch.load(path, map_location='cpu')
        print(f"load model from {path}")
        self.load_state_dict(ckpt['model'],strict=False)

    def forward(self,x,feat_old=None,alpha=0,radius=None):
        x = self.convnet(x).view(-1, self.encoder_dim)
        feature = self.projection_dim(x)
        feature = self.bn(feature)
        if self.configs['adversarial'] and self.training:
            reverse_feature_new = ReverseLayerF.apply(feature, alpha)
            model_out_new = self.discriminator(reverse_feature_new)
            model_out_old = self.discriminator(feat_old)

        if self.use_cls and self.training:
            out = self.projection_cls(feature)
            if self.configs['adversarial']:
                if self.configs['eboundary'] and radius is not None:
                    radius = self.eboundary(radius)
                    return feature,out,model_out_new,model_out_old,radius
                else:
                    return feature, out, model_out_new, model_out_old
            elif self.configs['eboundary'] and radius is not None:
                radius = self.eboundary(radius)
                return feature, out, radius
            else:
                return feature, out
        else:
            return feature


class CreateModelWithTransformation(nn.Module):
    def __init__(self,base_model,k=4):
        '''
        :param args: {"arch":,
                     "pretrained": true or false,
                     "pretrained_path":str or None,
                     "num_class":,
                     "use_cls":true or false,
                     "out_dim:,
                     "adversarial":False}
        '''
        super().__init__()
        self.base = base_model
        for param in self.base.parameters():
            param.requires_grad = False
        self.T = Transformation(K=k)


    def forward(self,x,wj=None):
        if self.training:
            feature,cls_close=self.base(x)
            feature = self.T(feature)
            wj = self.T(wj)
            return feature, wj
        else:
            feature = self.base(x)
            feature = self.T(feature)
            return feature


class CreateBaseReidModel(nn.Module):
    def __init__(self,configs,cls_config):
        '''
        :param args: {"arch":,
                     "pretrained": true or false,
                     "pretrained_path":str or None,
                     "num_class":,
                     "use_cls":true or false,
                     "out_dim:}
        '''
        super().__init__()
        self.cls_config = cls_config
        if configs['arch'] in timm.list_models(pretrained=True):
            self.convnet = timm.create_model(configs['arch'], pretrained=configs['pretrained'], num_classes=0)
        else:
            raise Exception("not supported model architecture")

        self.encoder_dim = self.convnet.num_features
        self.use_cls = configs['use_cls']
        self.out_dim = configs['out_dim']
        self.projection_dim = nn.Linear(self.encoder_dim, configs['out_dim'], bias=False)
        self.bn = nn.BatchNorm1d(configs['out_dim'])
        if configs['use_cls']:
            self.projection_cls = self.get_classifier(configs['out_dim'], configs['num_class'])
        if configs['pretrained_path'] is not None and osp.exists(configs['pretrained_path']):
            self.load_state(configs['pretrained_path'])


    def load_state(self, path):
        ckpt = torch.load(path, map_location='cpu')
        self.load_state_dict(ckpt['model'])

    def forward(self,x,**kwargs):
        # TODO, combined with camera_id and view_id
        x = self.convnet(x).view(-1, self.encoder_dim)
        feature = self.projection_dim(x)
        feature = self.bn(feature)

        if self.use_cls and self.training:
            if self.cls_config['loss_type']  in ('arcface', 'cosface', 'amsoftmax', 'circle'):
                cls_score = self.projection_cls(feature, kwargs['label'])
            else:
                cls_score = self.projection_cls(feature)
            return cls_score,feature
        else:
            return feature


    def get_classifier(self,in_planes,num_classes):
        if self.cls_config['loss_type'] == 'arcface':
            print('using {} with s:{}, m: {}'.format(self.cls_config['loss_type'], self.cls_config['cosine_scale'],
                                                     self.cls_config['cosine_margin']))
            classifier = Arcface(in_planes,num_classes,
                                      s=self.cls_config['cosine_scale'], m=self.cls_config['cosine_margin'])
        elif self.cls_config['loss_type'] == 'cosface':
            print('using {} with s:{}, m: {}'.format(self.cls_config['loss_type'], self.cls_config['cosine_scale'],
                                                     self.cls_config['cosine_margin']))
            classifier = Cosface(in_planes,num_classes,
                                      s=self.cls_config['cosine_scale'], m=self.cls_config['cosine_margin'])
        elif self.cls_config['loss_type'] == 'arcface' == 'amsoftmax':
            print('using {} with s:{}, m: {}'.format(self.cls_config['loss_type'], self.cls_config['cosine_scale'], self.cls_config['cosine_margin']))
            classifier = AMSoftmax(in_planes,num_classes,
                                        s=self.cls_config['cosine_scale'], m=self.cls_config['cosine_margin'])
        elif self.cls_config['loss_type'] == 'arcface' == 'circle':
            print('using {} with s:{}, m: {}'.format(self.cls_config['loss_type'], self.cls_config['cosine_scale'], self.cls_config['cosine_margin']))
            classifier = CircleLoss(in_planes,num_classes,
                                         s=self.cls_config['cosine_scale'], m=self.cls_config['cosine_margin'])
        else:
            classifier = nn.Linear(in_planes,num_classes, bias=False)
            classifier.apply(self.weights_init_classifier)
        return classifier

    @staticmethod
    def weights_init_classifier(m):
        classname = m.__class__.__name__
        if classname.find('Linear') != -1:
            nn.init.normal_(m.weight, std=0.001)
            if m.bias:
                nn.init.constant_(m.bias, 0.0)
