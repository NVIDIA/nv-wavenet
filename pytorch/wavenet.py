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
# 
# *****************************************************************************
import torch
import wavenet
import math

class Conv(torch.nn.Module):
    """
    A convolution with the option to be causal and use xavier initialization
    """
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=None, dilation=1, bias=True,
                 w_init_gain='linear', is_causal=False):
        super(Conv, self).__init__()
        self.is_causal = is_causal
        # Padding depends on if a causal convolution is needed
        # kernel = 1 makes causal == non-causal
        if padding is None:
            if is_causal:
                padding = int((kernel_size - 1) * (dilation))
            else:
                assert(kernel_size % 2 == 1)
                padding = int(dilation * (kernel_size - 1) / 2)

        self.conv = torch.nn.Conv1d(in_channels, out_channels,
                                    kernel_size=kernel_size, stride=stride,
                                    padding=padding, dilation=dilation,
                                    bias=bias)

        torch.nn.init.xavier_uniform(
            self.conv.weight, gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, signal):
        conv_signal = self.conv(signal)
        if self.is_causal:
            # Removing padding by-products at the end
            conv_signal = conv_signal[:, :, :signal.size(2)]
        return conv_signal


class WaveNet(torch.nn.Module):
    def __init__(self, n_in_channels, n_layers, max_dilation,
                 n_residual_channels, n_skip_channels, n_out_channels,
                 n_cond_channels, upsamp_window, upsamp_stride):
        super(WaveNet, self).__init__()

        self.upsample = torch.nn.ConvTranspose1d(n_cond_channels,
                                                 n_cond_channels,
                                                 upsamp_window,
                                                 upsamp_stride)
        
        self.n_layers = n_layers
        self.max_dilation = max_dilation
        self.n_residual_channels = n_residual_channels 
        self.n_out_channels = n_out_channels
        self.cond_layers = Conv(n_cond_channels, 2*n_residual_channels*n_layers,
                                w_init_gain='tanh')
        self.dilate_layers = torch.nn.ModuleList()
        self.res_layers = torch.nn.ModuleList()
        self.skip_layers = torch.nn.ModuleList()
        
        self.embed = torch.nn.Embedding(n_in_channels,
                                             n_residual_channels)
        self.conv_out = Conv(n_skip_channels, n_out_channels,
                                 bias=False, w_init_gain='relu')
        self.conv_end = Conv(n_out_channels, n_out_channels,
                                 bias=False, w_init_gain='linear')

        loop_factor = math.floor(math.log2(max_dilation)) + 1
        for i in range(n_layers):
            dilation = 2 ** (i % loop_factor)
            
            # Kernel size is 2 in nv-wavenet
            in_layer = Conv(n_residual_channels, 2*n_residual_channels,
                                kernel_size=2, dilation=dilation,
                                w_init_gain='tanh', is_causal=True)
            self.dilate_layers.append(in_layer)

            # last one is not necessary
            if i < n_layers - 1:
                res_layer = Conv(n_residual_channels, n_residual_channels,
                                     w_init_gain='linear')
                self.res_layers.append(res_layer)

            skip_layer = Conv(n_residual_channels, n_skip_channels,
                                  w_init_gain='relu')
            self.skip_layers.append(skip_layer)

    def forward(self, forward_input):
        features = forward_input[0]
        forward_input = forward_input[1]
        cond_input = self.upsample(features)

        assert(cond_input.size(2) >= forward_input.size(1))
        if cond_input.size(2) > forward_input.size(1):
            cond_input = cond_input[:, :, :forward_input.size(1)]
       
        forward_input = self.embed(forward_input.long())
        forward_input = forward_input.transpose(1, 2)
       
        cond_acts = self.cond_layers(cond_input)
        for i in range(self.n_layers):
            in_act = self.dilate_layers[i](forward_input)
            start = i*2*self.n_residual_channels
            end = (i+1)*2*self.n_residual_channels
            in_act = in_act + cond_acts[:,start:end,:]
            t_act = torch.nn.functional.tanh(in_act[:, :self.n_residual_channels, :])
            s_act = torch.nn.functional.sigmoid(in_act[:, self.n_residual_channels:, :])
            acts = t_act * s_act
            if i < len(self.res_layers):
                res_acts = self.res_layers[i](acts)
            forward_input = res_acts + forward_input

            if i == 0:
                output = self.skip_layers[i](acts)
            else:
                output = self.skip_layers[i](acts) + output

        output = torch.nn.functional.relu(output, True)
        output = self.conv_out(output)
        output = torch.nn.functional.relu(output, True)
        output = self.conv_end(output)

        # Remove last probabilities because they've seen all the data
        last = output[:, :, -1]
        last = last.unsqueeze(2)
        output = output[:, :, :-1]

        # Replace probability for first value with 0's because we don't know
        first = last * 0.0
        output = torch.cat((first, output), dim=2)

        return output

    def export_weights(self):
        """
        Returns a dictionary with tensors ready for nv_wavenet wrapper
        """
        model = {}
        # We're not using a convolution to start to this does nothing
        model["embedding_prev"] = torch.cuda.FloatTensor(self.n_out_channels,
                                              self.n_residual_channels).fill_(0.0)

        model["embedding_curr"] = self.embed.weight.data
        model["conv_out_weight"] = self.conv_out.conv.weight.data
        model["conv_end_weight"] = self.conv_end.conv.weight.data
        
        dilate_weights = []
        dilate_biases = []
        for layer in self.dilate_layers:
            dilate_weights.append(layer.conv.weight.data)
            dilate_biases.append(layer.conv.bias.data)
        model["dilate_weights"] = dilate_weights
        model["dilate_biases"] = dilate_biases
       
        model["max_dilation"] = self.max_dilation

        res_weights = []
        res_biases = []
        for layer in self.res_layers:
            res_weights.append(layer.conv.weight.data)
            res_biases.append(layer.conv.bias.data)
        model["res_weights"] = res_weights
        model["res_biases"] = res_biases
        
        skip_weights = []
        skip_biases = []
        for layer in self.skip_layers:
            skip_weights.append(layer.conv.weight.data)
            skip_biases.append(layer.conv.bias.data)
        model["skip_weights"] = skip_weights
        model["skip_biases"] = skip_biases
        
        model["use_embed_tanh"] = False
    
        return model

    def get_cond_input(self, features):
        """
        Takes in features and gets the 2*R x batch x # layers x samples tensor
        """
        # TODO(rcosta): trim conv artifacts. mauybe pad spec to kernel multiple
        cond_input = self.upsample(features)
        time_cutoff = self.upsample.kernel_size[0] - self.upsample.stride[0]
        cond_input = cond_input[:, :, :-time_cutoff]
        cond_input = self.cond_layers(cond_input).data
        cond_input = cond_input.view(cond_input.size(0), self.n_layers, -1, cond_input.size(2))
        # This makes the data channels x batch x num_layers x samples
        cond_input = cond_input.permute(2,0,1,3)
        return cond_input
