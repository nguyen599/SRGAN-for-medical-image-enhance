import tensorflow as tf
from tensorflow.keras.applications.vgg19 import VGG19, preprocess_input
with strategy.scope():
    layer_5_4 = 20
    vgg = VGG19(input_shape=(*HR_IMG_SIZE, 3), include_top=False)
    VGG = tf.keras.Model(vgg.input, vgg.layers[layer_5_4].output)
    VGG.trainable = False


@tf.function
def _adversarial_loss(d_model, loss_object, hr_fake):
    _, y_discrim_logits = d_model(hr_fake)
    return tf.reduce_mean(loss_object(y_discrim_logits, tf.ones_like(y_discrim_logits)))

@tf.function
def denormalize(x):
    return (x + 1) * 127.5

@tf.function
def gen_loss_fn(d_model, loss_fn1 = tf.keras.losses.BinaryCrossentropy(), loss_fn2 = tf.keras.losses.MeanSquaredError(), hr = None, hr_fake = None):
    feature_fake = VGG(preprocess_input(denormalize(hr_fake))) / 12.75
    feature_real = VGG(preprocess_input(denormalize(hr))) / 12.75
    g_gan_loss = 1e-3 * _adversarial_loss(d_model, loss_fn1, hr_fake) # adversarial loss
    mse_loss = loss_fn2(hr, hr_fake) # content loss
    vgg_loss = loss_fn2(feature_real, feature_fake)  # content loss
    return mse_loss, g_gan_loss, vgg_loss

@tf.function
def disc_loss_fn(loss_object = tf.keras.losses.BinaryCrossentropy(), y_real_pred_logits = None, y_fake_pred_logits = None):
    loss_real = tf.reduce_mean(loss_object(tf.ones_like(y_real_pred_logits), y_real_pred_logits))
    loss_fake = tf.reduce_mean(loss_object(tf.zeros_like(y_fake_pred_logits), y_fake_pred_logits))
    return loss_real + loss_fake

@tf.function
def res2img(img):
    return tf.round(denormalize(img))

@tf.function
def compute_psnr(original_image, generated_image):
    psnr = tf.image.psnr(res2img(original_image), res2img(generated_image), max_val=255)
    return tf.math.reduce_mean(psnr, axis=None, keepdims=False, name=None)

@tf.function
def compute_ssim(original_image, generated_image):
    ssim = tf.image.ssim(res2img(original_image), res2img(generated_image), max_val=255)
    return tf.math.reduce_mean(ssim, axis=None, keepdims=False, name=None)

class PSNR(tf.keras.metrics.Metric):

    def __init__(self, name='psnr', **kwargs):
        super(PSNR, self).__init__(name=name, **kwargs)
        self.total = self.add_weight(name='total', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')
        self.compute_psnr = compute_psnr

    def update_state(self, original_image, generated_image, sample_weight=None):

        psnr = self.compute_psnr(original_image, generated_image)
        self.total.assign_add(psnr)
        self.count.assign_add(1.0)

    @tf.function
    def result(self):
        if self.count != 0:
            return self.total / self.count
        else:
            return 0.0

    def reset_states(self):
        self.total.assign(0.0)
        self.count.assign(0.0)
