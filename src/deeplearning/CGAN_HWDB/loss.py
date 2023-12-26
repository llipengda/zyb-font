import torch
import torch.nn.functional as F

from torch import nn
from torch import autograd


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ALPHA = 3
ALPHA_GP = 10
BETA_D = 1
BETA_P = 0.2
BETA_R = 0.2
LAMBDA_LI = 50
LAMBDA_PHI = 75
PHI_P = 3
PHI_R = 5
EPS = 0.05


class LabelSmoothing(nn.Module):
    def __init__(self, smoothing=0.1):
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        logprobs = F.log_softmax(x, dim=-1)

        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, x_fake, x_real):
        N = x_real.size(0)
        smooth = 1

        input_flat = x_fake.view(N, -1)
        target_flat = x_real.view(N, -1)

        intersection = input_flat * target_flat
        loss = (
            2
            * (intersection.sum(1) + smooth)
            / (input_flat.sum(1) + target_flat.sum(1) + smooth)
        )
        loss = 1 - loss.sum() / N
        return loss


# refer to https://github.com/caogang/wgan-gp/blob/master/gan_mnist.py


def calc_gradient_penalty(D, x_real, x_fake, x1, x2):

    x_real.requires_grad = True
    x_fake.requires_grad = True
    x1.requires_grad = True
    x2.requires_grad = True
    alpha = torch.rand(x1.shape[0], 1, 1, 1)
    alpha = alpha.expand(x_real.size())
    alpha = alpha.to(DEVICE)

    interpolates = alpha * x_real + ((1 - alpha) * x_fake)

    interpolates = interpolates.to(DEVICE)

    disc_interpolates = D(interpolates, x1, x2)[:3]

    gradients = autograd.grad(
        outputs=disc_interpolates,
        inputs=[interpolates, x1, x2],
        grad_outputs=[
            torch.ones(disc_interpolates[0].size()).to(DEVICE),
            torch.ones(disc_interpolates[1].size()).to(DEVICE),
            torch.ones(disc_interpolates[2].size()).to(DEVICE),
        ],
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


class GenerationLoss(nn.Module):
    def __init__(self):
        super(GenerationLoss, self).__init__()
        self.cls_criteron = LabelSmoothing()

    def forward(
        self,
        out: list[torch.Tensor],
        out_real: list[torch.Tensor],
        features: list[torch.Tensor],
        features_real: list[torch.Tensor],
        real_label: torch.Tensor,
        real_style_label: torch.Tensor,
        char_label: torch.Tensor,
        x_fake: torch.Tensor,
        x_real: torch.Tensor,
        encoder_out_real_left: torch.Tensor,
        encoder_out_fake_left: torch.Tensor,
        encoder_out_real_right: torch.Tensor,
        encoder_out_fake_right: torch.Tensor,
        cls_enc_p: torch.Tensor | None = None,
        cls_enc_s: torch.Tensor | None = None,
    ):
        self.real_fake_loss = ALPHA * nn.BCELoss()(
            out[0], real_label.float().view(-1, 1)
        )
        self.style_category_loss = BETA_D * self.cls_criteron(
            out[1], real_style_label
        )
        self.char_category_loss = BETA_D * self.cls_criteron(
            out[2], char_label
        )

        self.reconstruction_loss = LAMBDA_LI * nn.L1Loss()(
            x_fake, x_real
        )

        assert len(features) == len(features_real)
        features = features[4:8]
        features_real = features_real[4:8]
        tmp_loss = 0
        for i in range(len(out[3])):
            tmp_loss += nn.MSELoss()(out[3][i], out_real[3][i])
        for i in range(len(features)):
            tmp_loss += nn.MSELoss()(features[i], features_real[i]) * EPS
        self.reconstruction_loss2 = LAMBDA_PHI * tmp_loss
        

        self.left_constant_loss = PHI_P * nn.MSELoss()(
            encoder_out_real_left, encoder_out_fake_left
        )
        
        self.right_constant_loss = PHI_R * nn.MSELoss()(
            encoder_out_real_right, encoder_out_fake_right
        )
        
        self.content_category_loss = BETA_P * self.cls_criteron(
            cls_enc_p, char_label
        )
        
        self.style_category_loss = BETA_R * self.cls_criteron(
            cls_enc_s, real_style_label
        )
        
        return (
            self.real_fake_loss
            + self.style_category_loss
            + self.char_category_loss
            + self.reconstruction_loss
            + self.reconstruction_loss2
            + self.left_constant_loss
            + self.right_constant_loss
            + self.content_category_loss
            + self.style_category_loss
        )


class DiscriminationLoss(nn.Module):
    def __init__(self):
        super(DiscriminationLoss, self).__init__()
        self.cls_criteron = LabelSmoothing()

    def forward(
        self,
        out_real,
        out_fake,
        real_label,
        fake_label,
        real_style_label,
        fake_style_label,
        char_label,
        fake_char_label,
        cls_enc_p=None,
        cls_enc_s=None,
        D=None,
        x_real=None,
        x_fake=None,
        x1=None,
        x2=None,
    ):
        self.real_loss = ALPHA * nn.BCELoss()(
            out_real[0], real_label.float().view(-1, 1)
        )  
        self.fake_loss = ALPHA * nn.BCELoss()(
            out_fake[0], fake_label.float().view(-1, 1)
        )  
        self.real_style_loss = BETA_D * self.cls_criteron(
            out_real[1], real_style_label
        ) 
        self.fake_style_loss = BETA_D * self.cls_criteron(
            out_fake[1], fake_style_label
        ) 
        self.real_char_category_loss = BETA_D * self.cls_criteron(
            out_real[2], char_label
        )  
        self.fake_char_category_loss = BETA_D * self.cls_criteron(
            out_fake[2], fake_char_label
        )  
        self.content_category_loss = BETA_P * self.cls_criteron(
            cls_enc_p, char_label
        )
        self.style_category_loss = BETA_R * self.cls_criteron(
            cls_enc_s, real_style_label
        )
        if D:
            self.gradient_penalty = ALPHA_GP * calc_gradient_penalty(
                D, x_real, x_fake, x1, x2
            )
        else:
            self.gradient_penalty = 0.0

        return 0.5 * (
            self.real_loss
            + self.fake_loss
            + self.real_style_loss
            + self.fake_style_loss
            + self.real_char_category_loss
            + self.fake_char_category_loss
            + self.content_category_loss
            + self.style_category_loss
            + self.gradient_penalty
        )
