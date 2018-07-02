# *****************************************************************************
#  Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#      * Redistributions of source code must retain the above copyright
#        notice, this list of conditions and the following disclaimer.
#      * Redistributions in binary form must reproduce the above copyright
#        notice, this list of conditions and the following disclaimer in the
#        documentation and/or other materials provided with the distribution.
#      * Neither the name of the NVIDIA CORPORATION nor the
#        names of its contributors may be used to endorse or promote products
#        derived from this software without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
#  ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
#  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#  DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
#  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
#  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
#  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
#  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
#  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# *****************************************************************************
import tensorflow as tf
import nv_wavenet_ext


def interleave_lists(a, b, c, d, e, f, g):
    return [x for t in zip(a, b, c, d, e, f, g) for x in t]


def column_major(x):
    """
    PyTorch Tensors are row major, so this just returns a transpose
    """
    assert(x.is_contiguous)
    if len(x.shape) == 1:
        return x

    if len(x.shape) == 3:
        assert(x.shape[2] == 1)
        x = tf.squeeze(x)

    if len(x.shape) == 2:
        return tf.transpose(x)

    if len(x.shape) == 4:
        return tf.permute(x, [3, 2, 1, 0])


def enum(**enums):
    return type('Enum', (), enums)


Impl = enum(AUTO=0, SINGLE_BLOCK=1, DUAL_BLOCK=2, PERSISTENT=3)


class NVWaveNet:
    def __init__(self, embedding_prev,
                 embedding_curr,
                 conv_out_weight,
                 conv_end_weight,
                 dilate_weights,
                 dilate_biases,
                 max_dilation,
                 res_weights,
                 res_biases,
                 skip_weights,
                 skip_biases,
                 use_embed_tanh):
        self.R = nv_wavenet_ext.num_res_channels()
        self.S = nv_wavenet_ext.num_skip_channels()
        self.A = nv_wavenet_ext.num_out_channels()

        self.max_dilation = max_dilation
        self.use_embed_tanh = use_embed_tanh
        assert embedding_prev.size() == (self.A, self.R), \
            ("embedding_prev: {} doesn't match compiled"
             " nv-wavenet size: {}").format(embedding_prev.size(),
                                            (self.A, self.R))
        self.embedding_prev = column_major(torch.t(embedding_prev))

        assert embedding_curr.size() == (self.A, self.R), \
            ("embedding_curr: {} doesn't match compiled"
             " nv-wavenet size: {}").format(embedding_curr.size(),
                                            (self.A, self.R))
        self.embedding_curr = column_major(torch.t(embedding_curr))

        assert conv_out_weight.size()[:2] == (self.A, self.S), \
            ("conv_out_weight: {} doesn't match compiled"
             " nv-wavenet size: {}").format(conv_out_weight.size()[:2],
                                            (self.A, self.S))
        self.conv_out = column_major(conv_out_weight)

        assert conv_end_weight.size()[:2] == (self.A, self.A), \
            ("conv_end_weight: {} doesn't match compiled"
             " nv-wavenet size: {}").format(conv_end_weight.size()[:2],
                                            (self.A, self.A))
        self.conv_end = column_major(conv_end_weight)

        dilate_weights_prev = []
        dilate_weights_curr = []
        for weight in dilate_weights:
            assert weight.size(2) == 2, \
                "nv-wavenet only supports kernel_size 2"
            assert weight.size()[:2] == (2*self.R, self.R), \
                ("dilated weight: {} doesn't match compiled"
                 " nv-wavenet size: {}").format(weight.size()[:2],
                                                (2*self.R, self.R))
            Wprev = column_major(weight[:, :, 0])
            Wcurr = column_major(weight[:, :, 1])
            dilate_weights_prev.append(Wprev)
            dilate_weights_curr.append(Wcurr)

        for bias in dilate_biases:
            assert(bias.size(0) == 2*self.R)
        for weight in res_weights:
            assert weight.size()[:2] == (self.R, self.R), \
                ("residual weight: {} doesn't match compiled"
                 " nv-wavenet size: {}").format(weight.size()[:2],
                                                (self.R, self.R))
        for bias in res_biases:
            assert(bias.size(0) == self.R), \
                ("residual bias: {} doesn't match compiled"
                 " nv-wavenet size: {}").format(bias.size(0), self.R)
        for weight in skip_weights:
            assert weight.size()[:2] == (self.S, self.R), \
                ("skip weight: {} doesn't match compiled"
                 " nv-wavenet size: {}").format(weight.size()[:2],
                                                (self.S, self.R))
        for bias in skip_biases:
            assert(bias.size(0) == self.S), \
                ("skip bias: {} doesn't match compiled"
                 " nv-wavenet size: {}").format(bias.size(0), self.S)

        dilate_biases = [column_major(bias) for bias in dilate_biases]
        res_weights = [column_major(weight) for weight in res_weights]
        res_biases = [column_major(bias) for bias in res_biases]
        skip_weights = [column_major(weight) for weight in skip_weights]
        skip_biases = [column_major(bias) for bias in skip_biases]

        # There's an extra residual layer that's not used
        res_weights.append(tf.zeros(self.R, self.R))
        res_biases.append(tf.zeros(self.R))

        assert(len(res_biases) == len(skip_biases) and
               len(res_biases) == len(dilate_biases) and
               len(res_weights) == len(skip_weights) and
               len(res_weights) == len(dilate_weights)), \
            """Number of layers is inconsistent for different parameter types.
        The list sizes should be the same for skip weights/biases and 
        dilate weights/biases.  Additionally the residual weights/biases
        lists should be one shorter.  But their sizes are:
        len(dilate_weights) = {}
        len(dilale_biases) = {}
        len(skip_weights) = {}
        len(skip_biases) = {}
        len(res_weights) = {}
        len(res_biases) = {}""".format(len(dilate_weights),
                                       len(dilate_biases),
                                       len(skip_weights),
                                       len(skip_biases),
                                       len(res_weights)-1,
                                       len(res_biases)-1)

        self.num_layers = len(res_biases)
        self.layers = interleave_lists(dilate_weights_prev,
                                       dilate_weights_curr,
                                       dilate_biases,
                                       res_weights,
                                       res_biases,
                                       skip_weights,
                                       skip_biases)

    def infer(self, cond_input, implementation):
        # cond_input is channels x batch x num_layers x samples
        assert(cond_input.shape[0:3:2] == (2*self.R, self.num_layers)), \
            """Inputs are channels x batch x num_layers x samples.
        Channels and num_layers should be sizes: {}
        But input is: {}""".format((2*self.R, self.num_layers),
                                   cond_input.shape[0:3:2])
        batch_size = cond_input.shape[1]
        sample_count = cond_input.shape[3]
        cond_input = column_major(cond_input)
        samples = tf.Tensor(batch_size, sample_count)
        nv_wavenet_ext.infer(samples,
                             sample_count,
                             batch_size,
                             self.embedding_prev,
                             self.embedding_curr,
                             self.conv_out,
                             self.conv_end,
                             cond_input,
                             self.num_layers,
                             self.use_embed_tanh,
                             self.max_dilation,
                             implementation,
                             *self.layers)
        return samples
