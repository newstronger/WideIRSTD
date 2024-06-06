import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

class spctral_residual(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    def forward(self,x):
        b,c,h,w=x.shape
        fm=[]
        for i in range(b):
            img=x[i,:,:,:]
            avg=torch.mean(img,dim=0)
            max=torch.max(img,dim=0)[0]
            out=(avg+max)
            fft=torch.fft.fft2(out)
            fshift=torch.fft.fftshift(fft)
            eps = 1e-10  # 一个很小的数
            amplitude_spectrum = torch.abs(fshift)
            amplitude_spectrum[amplitude_spectrum == 0] = eps
            LogAmp=torch.log(amplitude_spectrum)
            BlurAmp=mean_filter2d(LogAmp)
            phase=torch.angle(fshift)
            Spectral_res = LogAmp - BlurAmp
            ishift = torch.fft.ifftshift(torch.exp(Spectral_res+1j*phase))
            Res_ifft = torch.fft.ifft2(ishift)
            Res = torch.abs(Res_ifft).expand(c,h,w)
            fm.append(Res)
        out=torch.stack(fm,dim=0)
        return out.sigmoid()*x



def mean_filter2d(image, kernel_size=3):
    # 确保卷积核的尺寸是一个奇数
    assert kernel_size % 2 == 1, "Kernel size must be odd."
    # 计算卷积核的归一化权重
    weight = torch.ones(1, 1, kernel_size, kernel_size) / (kernel_size * kernel_size)
    weight = weight.cuda()
    # 对输入图像进行适当的填充以处理边缘像素
    padding = kernel_size // 2
    image = image.unsqueeze(0).unsqueeze(0)
    filtered_image = F.conv2d(image, weight, padding=padding)
    return filtered_image.squeeze(0).squeeze(0)

# 示例
if __name__ == "__main__":
    import cv2
    import numpy as np
    from matplotlib import pyplot as plt

    # 读取图像并转换为 PyTorch tensor
    # image_path = '/home/Newdisk/wangzhang/Infraed/NUDT-SIRST/images/001327.png'  # 替换为实际图片路径
    # image = cv2.imread(image_path)
    # gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    # image_tensor=torch.from_numpy(gray).unsqueeze(0).unsqueeze(0)
    # Res=spctral_residual()(image_tensor).numpy()
    # rge=np.max(Res)-np.min(Res)
    # norm=(Res-np.min(Res))*255/rge
    # print(norm.shape)
    # cv2.imwrite('sm1.jpg',norm)
    x=torch.randn(8,16,128,128)
    y=spctral_residual()(x)
    print(y.shape)



   