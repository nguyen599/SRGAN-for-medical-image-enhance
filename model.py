import tensorflow as tf
#Generator model
# @tf.keras.saving.register_keras_serializable()
class Generator(tf.keras.Model):
    def __init__(self, num_blocks = 16, **kwargs):
#         super(Generator, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), strides=1, padding='same')
        # self.prelu = tf.keras.layers.PReLU(shared_axes=[1,2])
        self.relu = tf.keras.layers.ReLU()

#         self.residual_blocks = []
#         for _ in tf.range(num_blocks):
#             self.residual_blocks.append(ResidualBlock())
#         self.residual_blocks = tf.keras.Sequential(self.residual_blocks)
        self.residual_blocks = tf.keras.Sequential([
            ResidualBlock(),
            ResidualBlock(),
            ResidualBlock(),
            ResidualBlock(),
            ResidualBlock(),
            ResidualBlock(),
            ResidualBlock(),
            ResidualBlock(),
            ResidualBlock(),
            ResidualBlock(),
            ResidualBlock(),
            ResidualBlock(),
            ResidualBlock(),
            ResidualBlock(),
            ResidualBlock(),
            ResidualBlock()
        ])
        self.conv2 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), strides=1, padding='same', bias_initializer = None)
        self.bn1 = tf.keras.layers.BatchNormalization(synchronized = True)

        self.conv3 = tf.keras.layers.Conv2D(filters=256, kernel_size=(3,3), strides=1, padding='same')
        self.subpixel1 = tf.keras.layers.Lambda(lambda x:tf.nn.depth_to_space(x, 2))
        # self.prelu1 = tf.keras.layers.PReLU(shared_axes=[1, 2])
        self.relu1 = tf.keras.layers.ReLU()

        self.conv4 = tf.keras.layers.Conv2D(filters=256, kernel_size=(3,3), strides=1, padding='same')
        self.subpixel2 = tf.keras.layers.Lambda(lambda x:tf.nn.depth_to_space(x, 2))
        # self.prelu2 = tf.keras.layers.PReLU(shared_axes=[1, 2])
        self.relu2 = tf.keras.layers.ReLU()

        self.conv5 = tf.keras.layers.Conv2D(filters=3, kernel_size=(1,1), strides=1, padding='same',  activation='tanh')

        self.input_layer = tf.keras.layers.Input(shape=(*LR_IMG_SIZE,3), name="input_layer")

        input_shape = self.input_layer
        outputs_shape = self.call(input_shape)

        super(Generator, self).__init__(inputs=input_shape, outputs=outputs_shape, **kwargs)

    def call(self, x):
        x = self.conv1(x)
        # x = self.prelu(x)
        x = self.relu(x)
        temp = x

        x = self.residual_blocks(x)
        x = self.conv2(x)
        x = self.bn1(x)
        x = x + temp

        x = self.conv3(x)
        x = self.subpixel1(x)
        # x = self.prelu1(x)
        x = self.relu1(x)

        x = self.conv4(x)
        x = self.subpixel2(x)
        # x = self.prelu2(x)
        x = self.relu2(x)

        x = self.conv5(x)

        return x

    def get_config(self):
        config = super(Generator, self).get_config()
        return config

    def summary_model(self, input_shape):
        inputs = tf.keras.Input(shape=input_shape)
        outputs = self.call(inputs)
        tf.keras.Model(inputs=inputs, outputs=outputs).summary()

class ResidualBlock(tf.keras.layers.Layer):
    def __init__(self):
        super(ResidualBlock, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(64, (3, 3), strides=(1, 1), padding='same')
        self.bn1 = tf.keras.layers.BatchNormalization(synchronized = True)
        # self.prelu = tf.keras.layers.PReLU(shared_axes=[1, 2])
        self.relu = tf.keras.layers.ReLU()
        self.conv2 = tf.keras.layers.Conv2D(64, (3, 3), strides=(1, 1), padding='same')
        self.bn2 = tf.keras.layers.BatchNormalization(synchronized = True)

    def call(self, x):
        z = self.conv1(x)
        z = self.bn1(z)
        # z = self.prelu(z)
        z = self.relu(z)
        z = self.conv2(z)
        z = self.bn2(z)
        x = x + z
        return x

#Discriminator model
# @tf.keras.saving.register_keras_serializable()
class Discriminator(tf.keras.Model):
    def __init__(self, learning_rate=1e-4, **kwargs):
#         super(Discriminator, self).__init__()

        self.learning_rate = learning_rate

        self.conv1 = tf.keras.layers.Conv2D(kernel_size=3, filters=64, strides=1, padding='same')
        self.leaky1 = tf.keras.layers.LeakyReLU(alpha=0.2)

        self.convblock = tf.keras.Sequential([
                                    ConvolutionBlock(64,  3, 2),
                                    ConvolutionBlock(128, 3, 2),
                                    ConvolutionBlock(128, 3, 2),
                                    ConvolutionBlock(256, 3, 2),
                                    ConvolutionBlock(256, 3, 2),
                                    ConvolutionBlock(512, 3, 1),
                                    ConvolutionBlock(512, 3, 1),
                                    ])
#         self.convblock = tf.keras.Sequential(self.convblock)

        self.flatten = tf.keras.layers.Flatten()
        self.Dense1 = tf.keras.layers.Dense(256)
        self.leaky2 = tf.keras.layers.LeakyReLU(alpha=0.2)
        self.Dense2 = tf.keras.layers.Dense(1)
        self.sigmoid = tf.keras.layers.Activation('sigmoid')

        self.input_layer = tf.keras.layers.Input(shape=(*HR_IMG_SIZE, 3), name="input_layer")

        input_shape = self.input_layer
        outputs_shape = self.call(input_shape)

        super(Discriminator, self).__init__(inputs=input_shape, outputs=outputs_shape, **kwargs)

    def call(self, x):
        x = self.conv1(x)
        x = self.leaky1(x)

        x = self.convblock(x)

        x = self.flatten(x)
        x = self.Dense1(x)
        x = self.leaky2(x)
        logits = self.Dense2(x)
        n = self.sigmoid(logits)
        return n, logits

    def get_config(self):
        config = super(Discriminator, self).get_config()
        return config

    def summary_model(self, input_shape):
        inputs = tf.keras.Input(shape=input_shape)
        outputs = self.call(inputs)
        tf.keras.Model(inputs=inputs, outputs=outputs).summary()

class ConvolutionBlock(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, strides):
        super(ConvolutionBlock, self).__init__()
        self.conv = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding='same')
        self.batch_norm = tf.keras.layers.BatchNormalization(synchronized=True)
        self.leaky_relu = tf.keras.layers.LeakyReLU(alpha=0.2)

    def call(self, x):
        x = self.conv(x)
        x = self.batch_norm(x)
        x = self.leaky_relu(x)
        return x