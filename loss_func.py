import torch
import torchvision.transforms.v2 as tr

CHANNELS_NUM = 3

def init_loss_fn(g_model, loss_fn, lr, hr):
    hr_fake = g_model(lr)
    return loss_fn(hr_fake, hr)

def _adversarial_loss(d_model, loss_object, hr_fake):
    y_discrim_logits = d_model(hr_fake.detach())
#     print(y_discrim_logits[0].shape)
    return loss_object(y_discrim_logits[0], torch.ones_like(y_discrim_logits[0]))

def renormalize_vgg(x):

    re_normalize = tr.Compose([tr.Normalize(mean=[-1 for _ in range(CHANNELS_NUM)], # invert norm
                                            std=[2 for _ in range(CHANNELS_NUM)]),
                                tr.Normalize(mean = [ 0.485, 0.456, 0.406 ], # renorm
                                             std = [ 0.229, 0.224, 0.225 ])]
                             )
    return re_normalize(x)

def gen_loss_fn(d_model, vgg, loss_fn1=torch.nn.BCEWithLogitsLoss(), loss_fn2=torch.nn.MSELoss(), hr=None, hr_fake=None):
    with torch.no_grad():
        feature_fake = vgg(renormalize_vgg(hr_fake)) / 12.75
        feature_real = vgg(renormalize_vgg(hr)) / 12.75
    g_gan_loss = 1e-3 * _adversarial_loss(d_model, loss_fn1, hr_fake) # adversarial loss
    mse_loss = loss_fn2(hr_fake, hr) # content loss
    vgg_loss = loss_fn2(feature_fake, feature_real) # content loss
    return mse_loss, g_gan_loss, vgg_loss

def disc_loss_fn(loss_object=torch.nn.BCEWithLogitsLoss(), y_real_pred_logits=None, y_fake_pred_logits=None):
    loss_real = loss_object(y_real_pred_logits, torch.ones_like(y_real_pred_logits))
    loss_fake = loss_object(y_fake_pred_logits, torch.zeros_like(y_fake_pred_logits))
    return loss_real + loss_fake