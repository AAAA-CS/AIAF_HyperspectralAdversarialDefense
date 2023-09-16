import torch
import torch.nn as nn
from torch.autograd import Variable


def get_act(name):
    if name == 'LeakyReLU':
        return nn.LeakyReLU(0.2)
    elif name == 'ReLU':
        return nn.ReLU()
    elif name == 'Tanh':
        return nn.Tanh()
    elif name == '':
        return None
    else:
        raise NameError('Unknown activation:'+name)

def drop_act(name):
    if name == 'DP':
        return nn.Dropout(0.5)
    elif name == '':
        return None
    else:
        raise NameError('Unknown activation:'+name)

def Pool_act(name):
    if name == 'MP':
        return nn.MaxPool2d(2)
    elif name == 'AP':
        return nn.AdaptiveMaxPool2d(1)
    elif name == '':
        return None
    else:
        raise NameError('Unknown activation:'+name)


# Reference: https://github.com/Alexander-H-Liu/UFDN/blob/master/src/ufdn.py
def LoadModel(name, parameter, img_size, input_dim,enc=None):
    if name == 'autoencoder':
        code_dim = parameter['code_dim']
        enc_list = []

        for layer,para in enumerate(parameter['encoder']):
            if para[0] == 'conv':
                if layer==0:
                    init_dim = input_dim
                next_dim,kernel_size,stride,pad,bn,act,pool,drop = para[1:9]
                act = get_act(act)
                pool = Pool_act(pool)
                drop = drop_act(drop)
                enc_list.append((para[0],(init_dim, next_dim,kernel_size,stride,pad,bn,act,pool,drop)))
                init_dim = next_dim
            else:
                raise NameError('Unknown encoder layer type:'+para[0])

        dec_list = []
        for layer,para in enumerate(parameter['decoder']):
            if para[0] == 'conv':
                next_dim,kernel_size,stride,pad,bn,act,insert_code = para[1:8]
                act = get_act(act)
                dec_list.append((para[0],(init_dim, next_dim,kernel_size,stride,pad,bn,act),insert_code))
                init_dim = next_dim
            else:
                raise NameError('Unknown decoder layer type:'+para[0])
        if enc == 1:
            return Denoiser(enc_list,dec_list,code_dim)
        if enc == 2:
            return Denoiser_special1(enc_list,code_dim)
        if enc == 3:
            return Denoiser_special2(enc_list,code_dim)
    elif name == 'nn':
        dnet_list = []
        init_dim = input_dim
        for para in parameter['dnn']:
            if para[0] == 'fc':
                next_dim,bn,act,dropout = para[1:5]
                act = get_act(act)
                dnet_list.append((para[0],(init_dim, next_dim,bn,act,dropout)))
                init_dim = next_dim
            else:
                raise NameError('Unknown nn layer type:'+para[0])
        return Discriminator(dnet_list)
    elif name == 'cnn':
        dnet_list = []
        init_dim = input_dim
        cur_img_size = img_size
        reshaped = False
        for layer,para in enumerate(parameter['dnn']):
            if para[0] == 'conv':
                next_dim,kernel_size,stride,pad,bn,act = para[1:7]
                act = get_act(act)
                dnet_list.append((para[0],(init_dim, next_dim,kernel_size,stride,pad,bn,act)))
                init_dim = next_dim
                cur_img_size = round(cur_img_size/2)
            elif para[0] == 'fc':
                if not reshaped:
                    init_dim = int(cur_img_size*cur_img_size*init_dim)
                    reshaped = True
                next_dim,bn,act,dropout = para[1:5]
                act = get_act(act)
                dnet_list.append((para[0],(init_dim, next_dim,bn,act,dropout)))
                init_dim = next_dim
            else:
                raise NameError('Unknown encoder layer type:'+para[0])
        return Discriminator(dnet_list)
    else:
        raise NameError('Unknown model type:'+name)


# custom weights initialization
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)


# create a Convolution/Deconvolution block
def ConvBlock(c_in, c_out, k=4, s=2, p=1, norm='bn', activation=None, pool_act=None,drop_act=None,transpose=False, dropout=None):
    layers = []
    if transpose:
        layers.append(nn.ConvTranspose2d(c_in, c_out, kernel_size=k, stride=s, padding=p))
    else:
        layers.append(nn.Conv2d(c_in, c_out, kernel_size=k, stride=s, padding=p))
    if dropout:
        layers.append(nn.Dropout2d(dropout))
    if norm == 'bn':
        layers.append(nn.BatchNorm2d(c_out))
    if activation is not None:
        layers.append(activation)
    if pool_act is not None:
        layers.append(pool_act)
    if drop_act is not None:
        layers.append(drop_act)
    return nn.Sequential(*layers)


# create a fully connected layer
def FC(c_in, c_out, norm='bn', activation=None, dropout=None):
    layers = []
    layers.append(nn.Linear(c_in,c_out))
    if dropout:
        if dropout>0:
            layers.append(nn.Dropout(dropout))
    if norm == 'bn':
        layers.append(nn.BatchNorm1d(c_out))
    if activation is not None:
        layers.append(activation)
    return nn.Sequential(*layers)


# Reference : https://github.com/pytorch/examples/blob/master/vae/main.py
class Denoiser(nn.Module):
    def __init__(self, enc_list, dec_list, attr_dim):
        super(Denoiser, self).__init__()

        # Encoder
        self.enc_layers = []
        self.fc1 = nn.Linear(40 * 8, 9)
        self.dropout = nn.Dropout(p=0.5)  # dropout训练

        for l in range(len(enc_list)):
            self.enc_layers.append(enc_list[l][0])
            if enc_list[l][0] == 'conv':
                c_in,c_out,k,s,p,norm,act,pool,drop = enc_list[l][1]
                if l == len(enc_list) :
                    setattr(self, 'enc_mu', ConvBlock(c_in,c_out,k,s,p,norm,act,pool,transpose=False))
                    setattr(self, 'enc_logvar', ConvBlock(c_in,c_out,k,s,p,norm,act,pool,transpose=False))
                else:
                    setattr(self, 'enc_'+str(l), ConvBlock(c_in,c_out,k,s,p,norm,act,pool,drop,transpose=False))
            elif enc_list[l][0] == 'fc':
                c_in,c_out,norm,act = enc_list[l][1]
                if l == len(enc_list) -1 :
                    setattr(self, 'enc_mu', FC(c_in,c_out,norm,act))
                    setattr(self, 'enc_logvar', FC(c_in,c_out,norm,act))
                else:
                    setattr(self, 'enc_'+str(l), FC(c_in,c_out,norm,act))
            else:
                raise ValueError('Unreconized layer type')

        # Decoder
        self.dec_layers = []
        self.attr_dim = attr_dim

        for l in range(len(dec_list)):
            self.dec_layers.append((dec_list[l][0],dec_list[l][2]))
            if dec_list[l][0] == 'conv':
                c_in,c_out,k,s,p,norm,act = dec_list[l][1]
                if dec_list[l][2]: c_in += self.attr_dim
                setattr(self, 'dec_'+str(l), ConvBlock(c_in,c_out,k,s,p,norm,act,transpose=True))
            elif dec_list[l][0] == 'fc':
                c_in,c_out,norm,act = dec_list[l][1]
                if dec_list[l][2]: c_in += self.attr_dim
                setattr(self, 'dec_'+str(l), FC(c_in,c_out,norm,act))
            else:
                raise ValueError('Unreconized layer type')

        self.apply(weights_init)

    def encode(self, x):
        for l in range(len(self.enc_layers)):
            if (self.enc_layers[l] == 'fc')  and (len(x.size())>2):
                batch_size = x.size()[0]
                x = x.view(batch_size,-1)
            x = getattr(self, 'enc_'+str(l))(x)

        return x

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z, insert_attrs = None):
        for l in range(len(self.dec_layers)):
            if (self.dec_layers[l][0] != 'fc') and (len(z.size()) != 4):
                z = z.unsqueeze(-1).unsqueeze(-1)
            if (insert_attrs is not None) and (self.dec_layers[l][1]):
                if len(z.size()) == 2:
                    z = torch.cat([z,insert_attrs],dim=1)
                else:
                    H,W = z.size()[2], z.size()[3]
                    z = torch.cat([z,insert_attrs.unsqueeze(-1).unsqueeze(-1).repeat(1,1,H,W)],dim=1)
            z = getattr(self, 'dec_'+str(l))(z)
        return z

    def forward(self,xadv1, xadv2, insert_attrs = None, return_enc = False):
        batch_size = xadv1.size()[0]
        #提取共有特征
        xadv1 = self.encode(xadv1)
        xadv2 = self.encode(xadv2)
        if len(xadv1.size()) > 2:
            xadv1 = xadv1.view(batch_size,-1)
            xadv2 = xadv2.view(batch_size,-1)
        z = (xadv1 + xadv2 )/2
        if return_enc:
            return z
        else:
            return self.decode(z,insert_attrs), xadv1, xadv2

class Denoiser_special1(nn.Module):
    def __init__(self, enc_list, attr_dim):
        super(Denoiser_special1, self).__init__()

        # Encoder
        self.enc_layers = []
        self.fc1 = nn.Linear(40 * 8, 9)
        self.dropout = nn.Dropout(p=0)  # dropout训练

        for l in range(len(enc_list)):
            self.enc_layers.append(enc_list[l][0])
            if enc_list[l][0] == 'conv':
                c_in,c_out,k,s,p,norm,act,pool,drop = enc_list[l][1]
                if l == len(enc_list) :
                    setattr(self, 'enc_mu', ConvBlock(c_in,c_out,k,s,p,norm,act,pool,transpose=False))
                    setattr(self, 'enc_logvar', ConvBlock(c_in,c_out,k,s,p,norm,act,pool,transpose=False))
                else:
                    setattr(self, 'enc_'+str(l), ConvBlock(c_in,c_out,k,s,p,norm,act,pool,drop,transpose=False))
            elif enc_list[l][0] == 'fc':
                c_in,c_out,norm,act = enc_list[l][1]
                if l == len(enc_list) -1 :
                    setattr(self, 'enc_mu', FC(c_in,c_out,norm,act))
                    setattr(self, 'enc_logvar', FC(c_in,c_out,norm,act))
                else:
                    setattr(self, 'enc_'+str(l), FC(c_in,c_out,norm,act))
            else:
                raise ValueError('Unreconized layer type')
        self.apply(weights_init)

    def encode(self, x):
        for l in range(len(self.enc_layers)):
            if (self.enc_layers[l] == 'fc')  and (len(x.size())>2):
                batch_size = x.size()[0]
                x = x.view(batch_size,-1)
            x = getattr(self, 'enc_'+str(l))(x)
        return x
    #传进来两个样本，但是只用到了第一个对抗样本的special
    def forward(self, xadv1, xadv2, insert_attrs = None, return_enc = False):
        batch_size = xadv1.size()[0]
        xadv1 = self.encode(xadv1)
        if len(xadv1.size()) > 2:
            xadv1 = xadv1.view(batch_size,-1)
        return xadv1

class Denoiser_special2(nn.Module):
    def __init__(self, enc_list, attr_dim):
        super(Denoiser_special2, self).__init__()

        # Encoder
        self.enc_layers = []
        self.fc1 = nn.Linear(40 * 8, 9)
        self.dropout = nn.Dropout(p=0.5)  # dropout训练

        for l in range(len(enc_list)):
            self.enc_layers.append(enc_list[l][0])
            if enc_list[l][0] == 'conv':
                c_in,c_out,k,s,p,norm,act,pool,drop = enc_list[l][1]
                if l == len(enc_list) :
                    setattr(self, 'enc_mu', ConvBlock(c_in,c_out,k,s,p,norm,act,pool,transpose=False))
                    setattr(self, 'enc_logvar', ConvBlock(c_in,c_out,k,s,p,norm,act,pool,transpose=False))
                else:
                    setattr(self, 'enc_'+str(l), ConvBlock(c_in,c_out,k,s,p,norm,act,pool,drop,transpose=False))
            elif enc_list[l][0] == 'fc':
                c_in,c_out,norm,act = enc_list[l][1]
                if l == len(enc_list) -1 :
                    setattr(self, 'enc_mu', FC(c_in,c_out,norm,act))
                    setattr(self, 'enc_logvar', FC(c_in,c_out,norm,act))
                else:
                    setattr(self, 'enc_'+str(l), FC(c_in,c_out,norm,act))
            else:
                raise ValueError('Unreconized layer type')

        self.apply(weights_init)

    def encode(self, x):
        for l in range(len(self.enc_layers)):
            if (self.enc_layers[l] == 'fc')  and (len(x.size())>2):
                batch_size = x.size()[0]
                x = x.view(batch_size,-1)
            x = getattr(self, 'enc_'+str(l))(x)
        return x

    # 传进来两个样本，但是只用到了第二个对抗样本的special
    def forward(self, xadv1, xadv2, insert_attrs = None, return_enc = False):
        batch_size = xadv2.size()[0]
        xadv2 = self.encode(xadv2)
        if len(xadv1.size()) > 2:
            xadv2 = xadv2.view(batch_size,-1)
        return xadv2

class Discriminator(nn.Module):
    def __init__(self, layer_list):
        super(Discriminator, self).__init__()

        self.layer_list = []

        for l in range(len(layer_list)-1):
            self.layer_list.append(layer_list[l][0])
            if layer_list[l][0] == 'conv':
                c_in,c_out,k,s,p,norm,act = layer_list[l][1]
                setattr(self, 'layer_'+str(l), ConvBlock(c_in,c_out,k,s,p,norm,act,transpose=False))
            elif layer_list[l][0] == 'fc':
                c_in,c_out,norm,act,drop = layer_list[l][1]
                setattr(self, 'layer_'+str(l), FC(c_in,c_out,norm,act,drop))
            else:
                raise ValueError('Unreconized layer type')

        self.layer_list.append(layer_list[-1][0])
        c_in,c_out,norm,act,_ = layer_list[-1][1]
        if not isinstance(c_out, list):
            c_out = [c_out]
        self.output_dim = len(c_out)

        for idx,d in enumerate(c_out):
            setattr(self, 'layer_out_'+str(idx), FC(c_in,d,norm,act,0))

        self.apply(weights_init)

    def forward(self, x):
        for l in range(len(self.layer_list)-1):
            if (self.layer_list[l] == 'fc') and (len(x.size()) != 2):
                batch_size = x.size()[0]
                x = x.view(batch_size,-1)
            x = getattr(self, 'layer_'+str(l))(x)

        output = []
        for d in range(self.output_dim):
            output.append(getattr(self,'layer_out_'+str(d))(x))

        if self.output_dim == 1:
            return output[0]
        else:
            return tuple(output)

# construct the network
OutChannel = 32
class CNN(nn.Module):
    def __init__(self, input_feature, Classes):
        super(CNN, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=input_feature,
                out_channels=OutChannel,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(OutChannel),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(OutChannel, OutChannel * 2, 3, 1, 1),
            nn.BatchNorm2d(OutChannel * 2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # nn.Dropout(0.5),

        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(OutChannel * 2, OutChannel * 4, 3, 1, 1),
            nn.BatchNorm2d(OutChannel * 4),
            nn.ReLU(),
            nn.AdaptiveMaxPool2d(1),
            # nn.Dropout(0.5),

        )

        self.out = nn.Linear(OutChannel * 4, Classes)  # fully connected layer, output 16 classes

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)  # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        x = self.out(x)
        return x