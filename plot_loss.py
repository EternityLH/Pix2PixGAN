import matplotlib.pyplot as plt

# 从loss_log中提取G_GAN和G_L1的数值
epochs = []
g_gan_values = []
g_l1_values = []

with open(r'D:\workspace\LLVIP-main\LLVIP-main\pix2pixGAN\checkpoints\KAIST_pix2pix\loss_log.txt', 'r') as f:
    lines = f.readlines()
    for line in lines:
        if 'G_GAN:' in line and 'G_L1:' in line:
            epoch = int(line.split('epoch: ')[1].split(',')[0])
            g_gan = float(line.split('G_GAN: ')[1].split(' ')[0])
            g_l1 = float(line.split('G_L1: ')[1].split(' ')[0])

            epochs.append(epoch)
            g_gan_values.append(g_gan)
            g_l1_values.append(g_l1)

# 绘制曲线
# plt.plot(epochs, g_gan_values, label='G_GAN')
plt.plot(epochs, g_l1_values, label='G_L1')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('KAIST_L1.png')
plt.show()