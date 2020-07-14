import torch
from torch import nn
from model_part import CreateUpConvBlock, DoubleConvolution
from pytorch_memlab import profile, MemReporter

class CreateUpConvBlockWithoutZ(nn.Module):
    def __init__(self, in_channel, concat_channel, mid_channel, out_channel, n=2, use_bn=True):
        super(CreateUpConvBlockWithoutZ, self).__init__()
        self.convTranspose = nn.ConvTranspose3d(in_channel, in_channel, (2, 2, 1), stride=(2, 2, 1), padding=(0, 0, 0), dilation=1)

        self.DoubleConvolution = DoubleConvolution(in_channel + concat_channel, mid_channel, out_channel, n=2, use_bn=True)

    def forward(self, x1, x2):
        x1 = self.convTranspose(x1)
        c = [(i - j) for (i, j) in zip(x2.size()[2:], x1.size()[2:])]
        x1 = nn.functional.pad(x1, (c[2] // 2, (c[2] * 2 + 1) // 2, c[1] // 2, (c[1] * 2 + 1) // 2, c[0] // 2, (c[0] * 2 + 1) // 2))

        x = torch.cat([x2, x1], dim=1)
        x = self.DoubleConvolution(x)
        return x

class PartOfUNet(nn.Module):
    def __init__(self, in_channel_up=128, in_channel_org=64, in_channel_half=128, num_class=14):
        super(PartOfUNet, self).__init__()

        self.expand_2 = CreateUpConvBlockWithoutZ(in_channel_up, in_channel_half, in_channel_half, in_channel_half)
        self.expand_1 = CreateUpConvBlockWithoutZ(in_channel_half, in_channel_org, in_channel_org, in_channel_org)
        
        self.segmentation = nn.Conv3d(in_channel_org, num_class, (1, 1, 1), stride=1, dilation=1, padding=(0, 0, 0))

        self.softmax = nn.Softmax(dim=1)

    @profile
    def forward(self, input_up, input_org, input_half):
        x = self.expand_2(input_up, input_half)
        x = self.expand_1(x, input_org)
        x = self.segmentation(x)
        x = self.softmax(x)

        return x


if __name__ == "__main__":
    in_channel_up = 64
    in_channel_org = 32
    in_channel_half = 64
    model = PartOfUNet(
            in_channel_up=in_channel_up,
            in_channel_org=in_channel_org,
            in_channel_half=in_channel_half,
            num_class=14
            )
    input_org_shape = [1, in_channel_org, 500, 500, 8]
    input_half_shape = [1, in_channel_half, 250, 250, 8]
    input_up_shape = [1, in_channel_up, 125, 125, 8]

    torch.backends.cudnn.enabled = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    reporter = MemReporter(model)
    reporter.report()    
    model.to(device)

    input_org_dummy = torch.rand(input_org_shape).to(device)
    input_half_dummy = torch.rand(input_half_shape).to(device)
    input_up_dummy = torch.rand(input_up_shape).to(device)

    print("Device:", device)
    print("Input:", input_org_dummy.shape, input_half_dummy.shape, input_up_dummy.shape)
    output = model(input_up_dummy, input_org_dummy, input_half_dummy)
    reporter.report()

    print("Output :", output.size())

