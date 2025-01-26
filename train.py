import numpy as np
from matplotlib import pylot as plt
import tensorflow as tf
from loss_func import *
from .model import Generator, Discriminator

with strategy.scope():
    g = Generator()
    d = Discriminator()

    g_optimizer = tf.keras.optimizers.Adam(1e-4)
    d_optimizer = tf.keras.optimizers.Adam(1e-4)

    generator_loss = gen_loss_fn
    discriminator_loss = disc_loss_fn
    loss_fn1 = tf.keras.losses.BinaryCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)
    loss_fn2 = tf.keras.losses.MeanSquaredError()
    psnr_calc = compute_psnr
    ssim_calc = compute_ssim

    g_loss_metric = tf.keras.metrics.Mean(name='g_loss')
    mse_loss_metric = tf.keras.metrics.Mean(name='mse_loss')
    g_gan_loss_metric = tf.keras.metrics.Mean(name='g_gan_loss')
    vgg_loss_metric = tf.keras.metrics.Mean(name='vgg_loss')
    d_loss_metric = tf.keras.metrics.Mean(name='d_loss')
    psnr_metric = tf.keras.metrics.Mean(name='psnr')
    ssim_metric = tf.keras.metrics.Mean(name='ssim')

with strategy.scope():

    @tf.function
    def train_step(image):
        lr, hr = image
        print(hr.shape)
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:

            # Generator
            fake_hr = g(lr, training = True)
            mse_loss, g_gan_loss, vgg_loss = generator_loss(d, loss_fn1, loss_fn2, hr, fake_hr)
            g_loss = mse_loss + g_gan_loss + vgg_loss

            # Discriminator
            d_hr_pred, d_hr_pred_logits = d(hr)
            d_hr_fake_pred, d_hr_fake_pred_logits = d(fake_hr)
            d_loss = discriminator_loss(loss_fn1, d_hr_pred_logits, d_hr_fake_pred_logits)
        generator_gradients = gen_tape.gradient(g_loss, g.trainable_variables)
        discriminator_gradients = disc_tape.gradient(d_loss, d.trainable_variables)

        g_optimizer.apply_gradients(zip(generator_gradients, g.trainable_variables))
        d_optimizer.apply_gradients(zip(discriminator_gradients, d.trainable_variables))


        psnr = psnr_calc(hr, fake_hr)
        ssim = ssim_calc(hr, fake_hr)

        d_loss_metric.update_state(d_loss)
        g_loss_metric.update_state(g_loss)
        mse_loss_metric.update_state(mse_loss)
        g_gan_loss_metric.update_state(g_gan_loss)
        vgg_loss_metric.update_state(vgg_loss)
        psnr_metric.update_state(psnr)
        ssim_metric.update_state(ssim)

        return d_loss_metric.result(),\
                g_loss_metric.result(),\
                mse_loss_metric.result(),\
                g_gan_loss_metric.result(),\
                vgg_loss_metric.result(), psnr_metric.result(), ssim_metric.result()

    @tf.function
    def distributed_train_step(dataset_inputs):
        per_replica_losses = strategy.run(train_step, args=(dataset_inputs,))
        x = []
        for loss in per_replica_losses:
            x.append(strategy.reduce(tf.distribute.ReduceOp.MEAN, loss, axis=None))
        x = tf.stack(x)
        return x

    @tf.function
    def distributed_train_epoch(train_dist_dataset):
        loss = tf.constant(0.0, dtype=tf.float32, shape=(7,))
        num_batches = 0.0
        for x in train_dist_dataset:
            loss += distributed_train_step(x)
            num_batches += 1.0
        loss = loss / num_batches
        return loss

#     @tf.function
    def distributed_test_epoch(g_model, test_dist_dataset, img_convert = True):

        permanent_sample = next(iter(test_dist_dataset))

        val_dataset_shuffled = test_dist_dataset.shuffle(buffer_size=len(hr_val_filenames))

        random_samples = val_dataset_shuffled.take(2)

        fig, ax = plt.subplots(3,3, figsize=(12,12))
        if img_convert:
            ax[0,0].imshow(np.round((permanent_sample[0][0]+1)*127.5).astype('uint8'))
            ax[0,1].imshow(np.round(((g_model(tf.expand_dims(permanent_sample[0][0], axis = 0))[0] + 1) * 127.5).numpy()).astype('uint8'))
            ax[0,2].imshow(np.round((permanent_sample[1][0]+1)*127.5).astype('uint8'))
            ax[0,0].set_yticklabels([])
            ax[0,1].set_yticklabels([])
            ax[0,2].set_yticklabels([])

            for i, batch in enumerate(random_samples):
                lr_batch, hr_batch = batch
                ax[i+1,0].imshow(np.round(((lr_batch[0]+1)*127.5)).astype('uint8'))
                ax[i+1,1].imshow(np.round(((g_model(tf.expand_dims(lr_batch[0], axis = 0))[0] + 1) * 127.5).numpy()).astype('uint8'))
                ax[i+1,2].imshow(np.round(((hr_batch[0]+1)*127.5)).astype('uint8'))
                ax[i+1,0].set_yticklabels([])
                ax[i+1,1].set_yticklabels([])
                ax[i+1,2].set_yticklabels([])
        else:
            ax[0,0].imshow(permanent_sample[0][0])
            ax[0,1].imshow(g_model(permanent_sample[0])[0])
            ax[0,2].imshow(permanent_sample[1][0])

            for i, batch in enumerate(random_samples):
                lr_batch, hr_batch = batch
                ax[i+1,0].imshow(lr_batch[0])
                ax[i+1,1].imshow(((g_model(lr_batch)[0])))
                ax[i+1,2].imshow(hr_batch[0])
        plt.subplots_adjust(wspace=0.05, hspace=0.05)
        plt.show()

        save_options = tf.saved_model.SaveOptions(experimental_io_device='/job:localhost')
        tf.saved_model.save(g,export_dir = '/kaggle/working/gen', options=save_options)
        tf.saved_model.save(d,export_dir = '/kaggle/working/disc', options=save_options)