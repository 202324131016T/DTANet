from model.model_block import *

class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.pool = nn.AdaptiveMaxPool1d(100)

        # k1: k-2p=1 (k,p,s)=(k=3,p=1,s=1)=(k=5,p=2,s=1)=(k=7,p=3,s=1) L->L/s=L/1
        # k1_: k-2p=s (k,p,s)=(k=4,p=1,s=2)=(k=6,p=2,s=2)=(k=8,p=3,s=2) L->L/s=L/2
        # kt: k+outp=2p+s (k,outp,p,s)=(k=4,outp=0,p=1,s=2)=(k=6,outp=0,p=2,s=2)=(k=8,outp=0,p=3,s=2) L->L*s=L*2
        k1, k2 = (7, 7) # s=1 L->L
        k1_, k2_ = (8, 7) # s=2 L->L/2
        kt, pt, st = (8, 3, 2) # ConvTranspose1d L->L*2
        self.resconv1_x = ResNet_basic_block0_simple(in_channels=128, out_channels=128, kernel_size=k1, kernel_size2=k2) # (N, C, L)->(N, C, L)
        self.resconv2_x = ResNet_basic_block1_simple(in_channels=128, out_channels=128, kernel_size=k1_, kernel_size2=k2_) # (N, C, L)->(N, C, L=50) L*0.5
        self.resconv3_x = ResNet_basic_block1_simple(in_channels=128, out_channels=128, kernel_size=k1_, kernel_size2=k2_) # (N, C, L)->(N, C, L=25) L*0.5
        # L->L 卷积最低L
        self.resconv1_x_ = ResNet_basic_block0_simple(in_channels=128, out_channels=128, kernel_size=k1, kernel_size2=k2)
        self.convt1_x = nn.ConvTranspose1d(in_channels=128, out_channels=128, kernel_size=kt, padding=pt, stride=st) # (N, C, L)->(N, C, L=50) L*2
        self.resconv4_x = ResNet_basic_block0_simple(in_channels=128, out_channels=128, kernel_size=k1, kernel_size2=k2) # (N, C, L)->(N, C, L)
        self.convt2_x = nn.ConvTranspose1d(in_channels=128, out_channels=128, kernel_size=kt, padding=pt, stride=st) # (N, C, L)->(N, C, L=100) L*2
        self.resconv5_x = ResNet_basic_block0_simple(in_channels=128, out_channels=300, kernel_size=k1, kernel_size2=k2) # (N, C, L)->(N, C, L)

        k1, k2 = (9, 9)
        k1_, k2_ = (10, 9)
        kt, pt, st = (10, 4, 2)
        self.resconv1_y = ResNet_basic_block0_simple(in_channels=128, out_channels=128, kernel_size=k1, kernel_size2=k2) # (N, C, L)->(N, C, L)
        self.resconv2_y = ResNet_basic_block1_simple(in_channels=128, out_channels=128, kernel_size=k1_, kernel_size2=k2_) # (N, C, L)->(N, C, L=50) L*0.5
        self.resconv3_y = ResNet_basic_block1_simple(in_channels=128, out_channels=128, kernel_size=k1_, kernel_size2=k2_) # (N, C, L)->(N, C, L=25) L*0.5
        # L->L 卷积最低L
        self.resconv1_y_ = ResNet_basic_block0_simple(in_channels=128, out_channels=128, kernel_size=k1, kernel_size2=k2)  # (N, C, L)->(N, C, L)
        self.convt1_y = nn.ConvTranspose1d(in_channels=128, out_channels=128, kernel_size=kt, padding=pt, stride=st) # (N, C, L)->(N, C, L=50) L*2
        self.resconv4_y = ResNet_basic_block0_simple(in_channels=128, out_channels=128, kernel_size=k1, kernel_size2=k2) # (N, C, L)->(N, C, L)
        self.convt2_y = nn.ConvTranspose1d(in_channels=128, out_channels=128, kernel_size=kt, padding=pt, stride=st) # (N, C, L)->(N, C, L=100) L*2
        self.resconv5_y = ResNet_basic_block0_simple(in_channels=128, out_channels=300, kernel_size=k1, kernel_size2=k2) # (N, C, L)->(N, C, L)

        # self.mhat1 = MHAT(n_head=2, d_k=128, d_v=64, d_x=128, d_o=128)  # 不共享参数，并行x，y
        # self.mhat2 = MHAT(n_head=2, d_k=128, d_v=64, d_x=128, d_o=128)  # 不共享参数，并行x，y
        self.mhat3 = MHAT(n_head=2, d_k=128, d_v=64, d_x=128, d_o=128)  # 不共享参数，并行x，y
        # self.mhat4 = MHAT(n_head=2, d_k=128, d_v=64, d_x=128, d_o=128)  # 不共享参数，并行x，y
        # self.mhat5 = MHAT(n_head=2, d_k=128, d_v=64, d_x=300, d_o=300)  # 不共享参数，并行x，y
        self.mhatx = MHAT(n_head=2, d_k=128, d_v=64, d_x=128, d_o=128)  # 不共享参数，并行x，y
        self.mhaty = MHAT(n_head=2, d_k=128, d_v=64, d_x=128, d_o=128)  # 不共享参数，并行x，y

    def forward(self, x, y):
        x = self.pool(x)

        # x(N=256, C=128, L=100)
        # y(N=256, C=128, L=1000)
        # x1,y1
        x1 = self.resconv1_x(x) # (N=256,  C=128, L=100)
        y1 = self.resconv1_y(y)  # (N=256, C=128, L=1000)
        # MHAT(x1,y1)
        # x1, y1 = self.mhat1(x1, y1)

        # x2,y2
        x2 = self.resconv2_x(x1)  # (N=256, C=128, L=50)
        y2 = self.resconv2_y(y1)  # (N=256, C=128, L=500)
        # MHAT(x2,y2)
        # x2, y2 = self.mhat2(x2, y2)
        # MHAT(x1,x2)(y1,y2)
        x1, x2 = self.mhatx(x1, x2)
        y1, y2 = self.mhaty(y1, y2)

        # x3,y3
        x3 = self.resconv3_x(x2) # (N=256, C=128, L=25)
        y3 = self.resconv3_y(y2) # (N=256, C=128, L=250)
        # x3_ y3_
        x3_ = self.resconv1_x_(x3) # (N=256, C=128, L=25)
        y3_ = self.resconv1_y_(y3) # (N=256, C=128, L=250)
        x3 = torch.add(x3, x3_)  # (N=256, C=128, L=25)
        y3 = torch.add(y3, y3_)  # (N=256, C=128, L=250)
        # MHAT(x3,y3)
        x3, y3 = self.mhat3(x3, y3)
        # MHAT(x1,x3)(y1,y3)
        # x1, x3 = self.mhatx(x1, x3)
        # y1, y3 = self.mhaty(y1, y3)
        # MHAT(x2,x3)(y2,y3)
        # x2, x3 = self.mhatx(x2, x3)
        # y2, y3 = self.mhaty(y2, y3)

        # add
        x3 = self.convt1_x(x3) # (N=256, C=128, L=50)
        x = torch.add(x3, x2)  # (N=256, C=128, L=50)
        y3 = self.convt1_y(y3) # (N=256, C=128, L=500)
        y = torch.add(y3, y2)  # (N=256, C=128, L=500)

        # x4,y4
        x4 = self.resconv4_x(x) # (N=256, C=128, L=50)
        y4 = self.resconv4_y(y) # (N=256, C=128, L=500)
        # MHAT(x4,y4)
        # x4, y4 = self.mhat4(x4, y4)
        # MHAT(x1,x4)(y1,y4)
        # x1, x4 = self.mhatx(x1, x4)
        # y1, y4 = self.mhaty(y1, y4)

        # add
        x4 = self.convt2_x(x4) # (N=256, C=128, L=100)
        x = torch.add(x4, x1)  # (N=256, C=128, L=100)
        y4 = self.convt2_y(y4) # (N=256, C=128, L=1000)
        y = torch.add(y4, y1)  # (N=256, C=128, L=1000)

        # x5,y5
        x5 = self.resconv5_x(x) # (N=256, C=300, L=100)
        y5 = self.resconv5_y(y) # (N=256, C=300, L=1000)
        # MHAT(x5,y5)
        # x5, y5 = self.mhat5(x5, y5)  # (N, C, L) (N=256, C=300, L=100)(N=256, C=300, L=1000)
        x, y = x5, y5

        return x, y # (N=256, C=300, L=100) (N=256, C=300, L=1000)

# input:x y  output:x y
class MHAT(nn.Module):
    def __init__(self, n_head=2, d_k=128, d_v=64, d_x=128, d_o=128):
        super(MHAT, self).__init__()
        # d_x:in_channels d_o:out_channels
        self.mhsa1 = MultiHead_SelfAttention(n_head=n_head, d_k=d_k, d_v=d_v, d_x=d_x, d_o=d_o)
        self.mhsa2 = MultiHead_SelfAttention(n_head=n_head, d_k=d_k, d_v=d_v, d_x=d_x, d_o=d_o)
        self.conv1 = nn.Conv1d(in_channels=d_o,
                               out_channels=d_o,
                               kernel_size=1,
                               stride=1,
                               padding=0)
        self.conv2 = nn.Conv1d(in_channels=d_o,
                               out_channels=d_o,
                               kernel_size=1,
                               stride=1,
                               padding=0)

    def forward(self, x, y):
        # (N,C,L) (256, 300, 100)(256, 300, 1000)
        x = x.permute(0, 2, 1) # (256, 100, 300)
        y = y.permute(0, 2, 1) # (256, 1000, 300)

        x_ = self.mhsa1(x, y) # (256, 100, 300)
        y_ = self.mhsa2(y, x) # (256, 1000, 300)

        x_ = x_.permute(0, 2, 1) # (256, 300, 100)
        y_ = y_.permute(0, 2, 1) # (256, 300, 1000)
        x = x.permute(0, 2, 1)  # (256, 300, 100)
        y = y.permute(0, 2, 1)  # (256, 300, 1000)

        # (N,C,L)
        x = self.conv1(x)
        y = self.conv2(y)

        x = torch.add(x, x_)
        y = torch.add(y, y_)

        return x, y

# input (256, 300) (256, 300)
class RegNet(nn.Module):
    def __init__(self):
        super(RegNet, self).__init__()

        self.pool = nn.AdaptiveMaxPool1d(1)  # pooling

        self.fc = nn.Sequential(
            nn.Linear(300 * 2, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

    def forward(self, x, y):
        # (256, 300, 128) (256, 300, 1280)
        x = self.pool(x) # (256, 300, 1)
        y = self.pool(y) # (256, 300, 1)

        x = x.squeeze() # (256, 300)
        y = y.squeeze() # (256, 300)

        r = torch.cat((x, y), dim=1) # (256, 300*2)

        r = self.fc(r) # (256, 1)

        r = r.squeeze() # (256)

        return r

class net(nn.Module):
    def __init__(self, FLAGS):
        super(net, self).__init__()
        self.embedding1 = nn.Embedding(FLAGS.charsmiset_size, 128)
        self.embedding2 = nn.Embedding(FLAGS.charseqset_size, 128)

        # replace extract feature layer
        self.resnet = ResNet()

        self.regnet = RegNet()

    def forward(self, x, y):
        # embedding
        x_init = Variable(x.long()).cuda() # (256, 100)
        x = self.embedding1(x_init) # (256, 100, 128)
        x_embedding = x.permute(0, 2, 1) # (256, 128, 100)
        # embedding
        y_init = Variable(y.long()).cuda() # (256, 1000)
        y = self.embedding2(y_init) # (256, 1000, 128)
        y_embedding = y.permute(0, 2, 1) # (256, 128, 1000)

        x, y = self.resnet(x_embedding, y_embedding) # (256, 300, 100) (256, 300, 1000)

        # input (256, 300, 100) (256, 300, 1000)
        out = self.regnet(x, y) # (256) 256个affinity
        # Reg block -> affinity

        return out # (256) 256个affinity


