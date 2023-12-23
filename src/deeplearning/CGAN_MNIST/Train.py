import os
import torch
import torchvision


from torch import Tensor, optim
from itertools import chain
from datetime import datetime
from torch.utils.data import DataLoader
from torchvision import transforms
from prefetch_generator import BackgroundGenerator
from torch.utils.tensorboard import writer


from deeplearning.CGAN_MNIST.CGAN_MNIST import CGAN_MNIST
from deeplearning.CGAN_MNIST.Module import CLSEncoderS, ClSEncoderP, Discriminator, Generater
from deeplearning.CGAN_MNIST.loss import DiscriminationLoss, GenerationLoss


class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


class Train:
    BATCH_SIZE = 48
    NUM_WORKERS = 8
    SAVE_INTERVAL = 100

    NUM_FONTS = 7
    NUM_CHARACTERS = 10

    BETA_1 = 0.5
    BETA_2 = 0.999
    INIT_LR_G = 0.0001
    INIT_LR_D = 0.001
    WEIGHT_DECAY = 0.001

    LAMBDA_L1 = 50

    def __init__(self, epochs=1201, reuse_path: str | None = None):
        self.__epochs = epochs
        self.__load_data()
        self.__device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        print("[INFO] Train - Using device:", self.__device)
        self.__reuse_path = reuse_path
        self.__init_module()
        log_dir = f'logs/CGAN_MNIST/{str(datetime.now()).replace(":", "-")}/'
        os.makedirs(log_dir, exist_ok=True)
        self.__writer = writer.SummaryWriter(log_dir)
        os.makedirs("out/CGAN_MNIST/", exist_ok=True)

    def __load_data(self):
        self.__transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
        ])
        self.__train_loader = DataLoaderX(
            CGAN_MNIST(transform=self.__transform),
            batch_size=Train.BATCH_SIZE,
            shuffle=True,
            num_workers=Train.NUM_WORKERS,
            pin_memory=True,
            # drop_last=True
        )

    def __init_module(self):
        self.__G = Generater().to(self.__device)
        self.__D = Discriminator(Train.NUM_FONTS + 1, Train.NUM_CHARACTERS + 1).to(
            self.__device
        )
        # 两个encoder后面的外接分类器
        self.__CLSP = ClSEncoderP(Train.NUM_CHARACTERS + 1).to(self.__device)
        self.__CLSS = CLSEncoderS(Train.NUM_FONTS + 1).to(self.__device)

        self.__optimizer_G = optim.Adam(
            self.__G.parameters(),
            lr=Train.INIT_LR_G,
            betas=(Train.BETA_1, Train.BETA_2),
            weight_decay=Train.WEIGHT_DECAY,
        )
        # scheduler_G = ExponentialLR(optimizer_G, 0.99)

        self.__optimizer_D = optim.Adam(
            chain(
                self.__D.parameters(),
                self.__CLSP.parameters(),
                self.__CLSS.parameters(),
            ),
            lr=Train.INIT_LR_D,
            betas=(Train.BETA_1, Train.BETA_2),
            weight_decay=Train.WEIGHT_DECAY,
        )
        
        if self.__reuse_path is not None:
            print(f"[INFO] Loading model from {self.__reuse_path}")
            # 增加font和char的数量不影响G的结构，所以G可以在不同试验中重复使用
            params = torch.load(self.__reuse_path, map_location="cpu")
            try:
                self.__G.load_state_dict(params["G"])
                self.__D.load_state_dict(params["D"])
                self.__CLSP.load_state_dict(params["CLSP"])
                self.__CLSS.load_state_dict(params["CLSS"])
                # 如果类别变化了，optimizer就算加载成功也会在step处报错
                self.__optimizer_G.load_state_dict(params["optimizer_G"])
                self.__optimizer_D.load_state_dict(params["optimizer_D"])
                print("[INFO] Loading success")
            except:
                pass

    def train(self, epoch: int):
        epoch_reconstruction_loss = 0.0
        epoch_lgs = 0.0
        epoch_lds = 0.0

        assert isinstance(self.__train_loader.dataset, CGAN_MNIST)

        x_fake: torch.Tensor | None = None
        x_real: torch.Tensor | None = None
        x1: torch.Tensor | None = None

        for (
            protype_img,
            protype_index,
            style_img,
            style_index,
            character_index,
            real_img
        ) in self.__train_loader:
            protype_img: Tensor = protype_img.to(self.__device)
            protype_index: Tensor = protype_index.to(self.__device)
            style_img: Tensor = style_img.to(self.__device)
            style_index: Tensor = style_index.to(self.__device)
            character_index: Tensor = character_index.to(self.__device)
            real_img: Tensor = real_img.to(self.__device)

            print(f"[INFO] Train - Epoch: {epoch}")

            x1 = protype_img
            x2 = style_img
            x_real = real_img
            real_style_label = style_index  # 真实的风格标签
            fake_style_label = torch.tensor(
                [
                    Train.NUM_FONTS for _ in range(x1.shape[0])
                ]  # drop_last = True的时候可以把用range(conf.batch_size)
            ).to(
                self.__device
            )  # 假的风格标签
            char_label = protype_index  # 真实的字形标签
            fake_char_label = torch.tensor(
                [Train.NUM_CHARACTERS for _ in range(x1.shape[0])]
            ).to(
                self.__device
            )  # 假的字形标签
            real_label = torch.tensor([1 for _ in range(x1.shape[0])]).to(
                self.__device
            )  # 真样本标签
            fake_label = torch.tensor([0 for _ in range(x1.shape[0])]).to(
                self.__device
            )  # 假样本标签

            self.__optimizer_G.zero_grad()
            x_fake, lout, rout = self.__G(x1, x2)
            out = self.__D(x_fake, x1, x2)
            out_real_ = self.__D(x_real, x1, x2)  # 加个下划线，避免跟后面重名

            # 两边encoder之后接一个分类器
            cls_enc_p = self.__CLSP(lout.view(-1, 512))
            cls_enc_s = self.__CLSS(rout.view(-1, 512))

            encoder_out_real_left = self.__G.left(x_real)[5]
            encoder_out_real_right = self.__G.right(x_real)[5]
            encoder_out_fake_left = self.__G.left(x_fake)[5]
            encoder_out_fake_right = self.__G.right(x_fake)[5]
            criterion_G = GenerationLoss()
            L_G = criterion_G(
                out,
                out_real_,
                real_label,
                real_style_label,
                char_label,
                x_fake,
                x_real,
                encoder_out_real_left,
                encoder_out_fake_left,
                encoder_out_real_right,
                encoder_out_fake_right,
                cls_enc_p,
                cls_enc_s,
            )
            epoch_reconstruction_loss += (
                criterion_G.reconstruction_loss.item() / Train.LAMBDA_L1
            )
            epoch_lgs += L_G.item()

            L_G.backward(retain_graph=True)  # 为何为提示重复backward
            self.__optimizer_G.step()

            self.__optimizer_D.zero_grad()
            assert x_fake is not None
            out_real = self.__D(x_real, x1, x2)
            out_fake = self.__D(x_fake.detach(), x1, x2)
            cls_enc_p = self.__CLSP(lout.view(-1, 512).detach())
            cls_enc_s = self.__CLSS(rout.view(-1, 512).detach())

            # 真假分类损失，风格分类损失，两个encoder提取到的特征质量损失
            L_D = DiscriminationLoss()(
                out_real,
                out_fake,
                real_label,
                fake_label,
                real_style_label,
                fake_style_label,
                char_label,
                fake_char_label,
                cls_enc_p,
                cls_enc_s,
                self.__D,
                x_real,
                x_fake.detach(),
                x1,
                x2,
            )
            epoch_lds += L_D.item()
            L_D.backward()
            self.__optimizer_D.step()

        epoch_reconstruction_loss /= len(self.__train_loader.dataset)
        epoch_lgs /= len(self.__train_loader.dataset)
        epoch_lds /= len(self.__train_loader.dataset)

        assert x_fake is not None and x_real is not None and x1 is not None

        fake_image = torchvision.utils.make_grid(x_fake)
        real_image = torchvision.utils.make_grid(x_real)
        src_image = torchvision.utils.make_grid(x1)
        self.__writer.add_image("img/fake", fake_image, epoch)
        self.__writer.add_image("img/real", real_image, epoch)
        self.__writer.add_image("img/src", src_image, epoch)
        self.__writer.add_scalars(
            "losses",
            {
                "G_LOSS": epoch_lgs,
                "D_LOSS": epoch_lds,
                "train_reconstruction": epoch_reconstruction_loss,
            },
            epoch,
        )
        if epoch % Train.SAVE_INTERVAL == 0:
            time_str = str(datetime.now()).replace(':', '-')
            out_path = f"out/CGAN_MNIST/{epoch}-{time_str}.pth"
            torch.save(
                {
                    "G": self.__G.state_dict(),
                    "D": self.__D.state_dict(),
                    "CLSP": self.__CLSP.state_dict(),
                    "CLSS": self.__CLSS.state_dict(),
                    "optimizer_G": self.__optimizer_G.state_dict(),
                    "optimizer_D": self.__optimizer_D.state_dict(),
                },
                out_path,
            )

    def __call__(self):
        for epoch in range(self.__epochs):
            self.train(epoch)


if __name__ == '__main__':
    train = Train(2401)
    train()
