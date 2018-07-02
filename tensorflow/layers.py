import tensorflow as tf

class CausalConv1D(tf.layers.Conv1D):
    def __init__(self, filters,
                 kernel_size,
                 strides=1,
                 dilation_rate=1,
                 activation=None,
                 use_bias=True,
                 kernel_initializer=tf.glorot_uniform_initializer(),
                 bias_initializer=None,
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 trainable=True,
                 name=None,
                 **kwargs):
        super(CausalConv1D, self).__init__(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding='valid',
            data_format='channels_first',
            dilation_rate=dilation_rate,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            trainable=trainable,
            name=name,
            **kwargs
        )
       
    def call(self, inputs):
        padding = (self.kernel_size[0] - 1) * self.dilation_rate[0]
        inputs = tf.pad(inputs, tf.constant(
            [(0, 0,), (0, 0), (1, 0)]) * padding)

        return super(CausalConv1D, self).call(inputs)

class Conv1DTranspose(tf.layers.Layer):
    def __init__(self, filters,
                 kernel_size=1,
                 strides=1,
                 trainable=True,
                 name=None,
                 **kwargs):
        super(Conv1DTranspose, self).__init__(
            trainable=trainable,
            name=name, **kwargs
        )
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides

    def call(self, inputs):
        inputs = tf.expand_dims(inputs, 0)
        inputs = tf.layers.conv2d_transpose(inputs,
                                          self.filters,
                                          (1,self.kernel_size),
                                          (1, self.strides)
                                          )
        inputs = tf.squeeze(inputs, 0)
        return super(Conv1DTranspose, self).call(inputs)
