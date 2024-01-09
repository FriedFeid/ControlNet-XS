from regex import D
import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import pytorch_lightning as pl
from contextlib import contextmanager, redirect_stdout

from ldm.modules.diffusionmodules.util import (
    conv_nd,
    linear,
    zero_module,
    timestep_embedding,
    checkpoint
    )

import ldm.modules.diffusionmodules.model as ae_model
from ldm.modules.distributions.distributions import DiagonalGaussianDistribution
from ldm.modules.diffusionmodules.model import Encoder, Decoder
from ldm.modules.ema import LitEma

from einops import rearrange
from ldm.modules.attention import BasicTransformerBlock, SpatialTransformer
from ldm.modules.diffusionmodules.openaimodel import (
    UNetModel,
    TimestepEmbedSequential,
    ResBlock as ResBlock_orig,
    Downsample,
    Upsample,
    AttentionBlock,
    TimestepBlock
    )
from ldm.util import exists, instantiate_from_config


class CrossAttInfusion(nn.Module):
    def __init__(
        self, in_channels, context_dim, n_heads=16, d_head=4,
        disable_self_attn=True, use_linear=True
    ):
        super().__init__()
        inner_dim = n_heads * d_head
        self.norm_infusion = Normalize(context_dim)
        self.use_linear = use_linear

        if not use_linear:
            self.proj_in_context = nn.Conv2d(
                context_dim,
                inner_dim,
                kernel_size=1,
                stride=1,
                padding=0
            )
        else:
            self.proj_in_context = nn.Linear(context_dim, inner_dim)
        
        self.attention = SpatialTransformer(
            in_channels=in_channels, depth=1, n_heads=n_heads, d_head=d_head,
            context_dim=inner_dim, disable_self_attn=disable_self_attn, use_linear=use_linear
        )

    def forward(self, x, infusion):
        infusion = self.norm_infusion(infusion)
        if not self.use_linear:
            infusion = self.proj_in_context(infusion)
        infusion = rearrange(infusion, 'b c h w -> b (h w) c').contiguous()
        if self.use_linear:
            infusion = self.proj_in_context(infusion)

        x = self.attention(x, infusion)
        return x


class MultiControlWrapper(nn.Module):
    def __init__(
            self,
            control_nets,
            model_channels,
    ):
        super().__init__()

        self.control_nets = nn.ModuleDict(control_nets)
        self.control_scales = {key: 1 for key in control_nets}
        self.model_channels = model_channels

    def forward(self, x, hint, timesteps, context, base_model, **kwargs):
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)

        embs = {key: None for key in self.control_scales}

        for key in self.control_scales:
            if self.control_nets[key].learn_embedding:
                embs[key] = self.control_nets[key].control_model.time_embed(t_emb)
            else:
                embs[key] = base_model.time_embed(t_emb)

        emb_base = base_model.time_embed(t_emb)
        base_scale = 1. - np.sum([self.control_scales[key] for key in self.control_scales])
        control_emb = 0
        for key in self.control_scales:
            control_emb = control_emb + self.control_scales[key] * embs[key]
        emb_base = emb_base * base_scale + control_emb

        guided_hints = {}
        for key in hint:
            guided_hints[key] = self.control_nets[key].input_hint_block(hint[key], embs[key], context)
            print('guided_hint_mean:', guided_hints[key].mean())

        h_base = x.type(base_model.dtype)
        h_ctrs = {key: x.type(base_model.dtype) for key in self.control_nets}
        hs_base = []
        hs_ctr = {key: [] for key in self.control_nets}
        it_enc_convs_in = {key: iter(self.control_nets[key].enc_zero_convs_in) for key in self.control_scales}
        it_enc_convs_out = {key: iter(self.control_nets[key].enc_zero_convs_out) for key in self.control_scales}
        it_dec_convs_in = {key: iter(self.control_nets[key].dec_zero_convs_in) for key in self.control_scales}
        it_dec_convs_out = {key: iter(self.control_nets[key].dec_zero_convs_out) for key in self.control_scales}

        ###################### Cross Control        ######################
        # input blocks (encoder)
        modules_inputBlock_ctr = {key: iter(self.control_nets[key].control_model.input_blocks) for key in self.control_nets}
        for nummer, module_base in enumerate(base_model.input_blocks):
            modules_ctr = {key: next(modules_inputBlock_ctr[key]) for key in modules_inputBlock_ctr}

            h_base = module_base(h_base, emb_base, context)
            h_ctrs = {key: modules_ctr[key](h_ctrs[key], embs[key], context) for key in modules_ctr}

            if guided_hints is not None:
                h_ctrs = {key: h_ctrs[key] + guided_hints[key] for key in h_ctrs}
                guided_hints = None

            # if not self.debugger['skip_enc_infusion']:
            #     if self.guiding in ('encoder_double', 'full'):
            correction = 0
            for key in self.control_scales:
                correction = correction + self.control_scales[key] * next(it_enc_convs_out[key])(h_ctrs[key], embs[key])
            h_base = h_base + correction

            hs_base.append(h_base)
            for key in hs_ctr:
                hs_ctr[key].append(h_ctrs[key])

            for key in self.control_nets:
                if self.control_nets[key].infusion2control == 'add':
                    h_ctrs[key] = h_ctrs[key] + next(it_enc_convs_in[key])(h_base, embs[key])
                elif self.control_nets[key].infusion2control == 'cat':
                    h_ctrs[key] = th.cat([h_ctrs[key], next(it_enc_convs_in[key])(h_base, embs[key])], dim=1)

        # mid blocks (bottleneck)
        h_base = base_model.middle_block(h_base, emb_base, context)
        for key in self.control_nets:
            h_ctrs[key] = self.control_nets[key].control_model.middle_block(h_ctrs[key], embs[key], context)

        correction = 0
        for key in self.control_nets.keys():
            correction = correction + self.control_scales[key] * self.control_nets[key].middle_block_out(h_ctrs[key], embs[key])
        h_base = h_base + correction

        for key in self.control_nets:
            if self.control_nets[key].guiding == 'full':
                raise NotImplementedError()
        # if self.guiding == 'full':
        #     for key in self.control_nets:
        #         if self.infusion2control == 'add':
        #             h_ctrs[key] = h_ctrs[key] + self.control_nets[key].middle_block_in(h_base, embs[key])
        #         elif self.infusion2control == 'cat':
        #             h_ctrs[key] = th.cat([h_ctrs[key], self.control_nets[key].middle_block_in(h_base, embs[key])], dim=1)

        # output blocks (decoder)
        modules_outputBlock_ctr = dict()
        for key in self.control_nets:
            modules_outputBlock_ctr[key] = self.control_nets[key].control_model.output_blocks if hasattr(
                self.control_nets[key].control_model, 'output_blocks') else [None] * len(base_model.output_blocks)

        for module_base in base_model.output_blocks:
            modules_ctr = {key: modules_outputBlock_ctr[key] for key in modules_outputBlock_ctr}

            correction = 0
            for key in self.control_nets:
                correction = correction + self.control_scales[key] * next(it_dec_convs_out[key])(hs_ctr[key].pop(), embs[key])

            h_base = h_base + correction

            h_base = th.cat([h_base, hs_base.pop()], dim=1)
            h_base = module_base(h_base, emb_base, context)

            ##### Quick and dirty attempt of fixing "full" with not applying corrections to the last layer #####
            for key in self.control_nets:
                if self.control_nets[key].guiding == 'full':
                    raise NotImplementedError()
            # if not self.debugger['skip_dec_infusion']:
            #     if self.guiding == 'full':
            #         h_ctr = th.cat([h_ctr, hs_ctr.pop()], dim=1)
            #         h_ctr = module_ctr(h_ctr, emb, context)
            #         if module_base != base_model.output_blocks[-1]:
            #             if self.infusion2base == 'add':
            #                 h_base = h_base + next(it_dec_convs_out)(h_ctr, emb)
            #             elif self.infusion2base == 'cat':
            #                 raise NotImplementedError()

            #             if self.infusion2control == 'add':
            #                 h_ctr = h_ctr + next(it_dec_convs_in)(h_base, emb)
            #             elif self.infusion2control == 'cat':
            #                 h_ctr = th.cat([h_ctr, next(it_dec_convs_in)(h_base, emb)], dim=1)

        return base_model.out(h_base)


class TwoStreamControlNet(nn.Module):
    def __init__(
            self,
            image_size,
            in_channels,
            model_channels,
            out_channels,
            hint_channels,
            num_res_blocks,
            attention_resolutions,
            dropout=0,
            channel_mult=(1, 2, 4, 8),
            conv_resample=True,
            dims=2,
            use_checkpoint=False,
            use_fp16=False,
            num_heads=-1,
            num_head_channels=-1,
            num_heads_upsample=-1,
            use_scale_shift_norm=False,
            resblock_updown=False,
            use_new_attention_order=False,
            use_spatial_transformer=False,  # custom transformer support
            transformer_depth=1,  # custom transformer support
            context_dim=None,  # custom transformer support
            n_embed=None,  # custom support for prediction of discrete ids into codebook of first stage vq model
            legacy=False,
            # disable_self_attentions=None,
            # num_attention_blocks=None,
            # disable_middle_self_attn=False,
            use_linear_in_transformer=False,
            infusion2control='cat',         # how to infuse intermediate information into the control net? {'add', 'cat', None}
            infusion2base='add',            # how to infuse intermediate information into the base net? {'add', 'cat', 'att'}
            guiding='encoder',              # use just encoder for control or the whole encoder + decoder net? {'encoder', 'encoder_double', 'full'}
            two_stream_mode='cross',        # mode for the two stream infusion. {'cross', 'sequential'}
            control_model_ratio=1.0,        # ratio of the control model size compared to the base model. [0, 1]
            debugger={
                'skip_enc_infusion': False,
                'skip_dec_infusion': False,
                'control_scale': 1.0
            },
            learn_embedding=True,
            n_heads=4,
            d_head=16,
            fixed=False,
    ):
        assert infusion2control in ('cat', 'add', 'att', None), f'infusion2control needs to be cat, add, att or None, but not {infusion2control}'
        assert infusion2base in ('add', 'att'), f'infusion2base only defined for add and att, but not {infusion2base}'
        assert guiding in ('encoder', 'encoder_double', 'full'), f'guiding has to be encoder, encoder_double or full, but not {guiding}'
        assert two_stream_mode in ('cross', 'sequential'), f'two_stream_mode has to be either cross or sequential, but not {two_stream_mode}'

        super().__init__()

        self.learn_embedding = learn_embedding
        self.infusion2control = infusion2control
        self.infusion2base = infusion2base
        self.in_ch_factor = 1 if infusion2control in ('add', 'att') else 2
        self.guiding = guiding
        self.two_stream_mode = two_stream_mode
        self.control_model_ratio = control_model_ratio
        self.out_channels = out_channels
        self.dims = 2
        self.model_channels = model_channels
        self.n_heads = n_heads
        self.d_head = d_head
        self.fixed = fixed

        self.debugger = debugger

        # with redirect_stdout(None):
        if True:
            ################# start control model variations #################
            base_model = UNetModel(
                image_size=image_size, in_channels=in_channels, model_channels=model_channels,
                out_channels=out_channels, num_res_blocks=num_res_blocks,
                attention_resolutions=attention_resolutions, dropout=dropout, channel_mult=channel_mult,
                conv_resample=conv_resample, dims=dims, use_checkpoint=use_checkpoint,
                use_fp16=use_fp16, num_heads=num_heads, num_head_channels=num_head_channels,
                num_heads_upsample=num_heads_upsample, use_scale_shift_norm=use_scale_shift_norm,
                resblock_updown=resblock_updown, use_new_attention_order=use_new_attention_order,
                use_spatial_transformer=use_spatial_transformer, transformer_depth=transformer_depth,
                context_dim=context_dim, n_embed=n_embed, legacy=legacy,
                # disable_self_attentions=disable_self_attentions,
                # num_attention_blocks=num_attention_blocks,
                # disable_middle_self_attn=disable_middle_self_attn,
                use_linear_in_transformer=use_linear_in_transformer,
                )  # initialise control model from base model

            if fixed:
                self.control_model = ControlledUNetModelFixed(
                    image_size=image_size, in_channels=in_channels, model_channels=model_channels,
                    out_channels=out_channels, num_res_blocks=num_res_blocks,
                    attention_resolutions=attention_resolutions, dropout=dropout, channel_mult=channel_mult,
                    conv_resample=conv_resample, dims=dims, use_checkpoint=use_checkpoint,
                    use_fp16=use_fp16, num_heads=num_heads, num_head_channels=num_head_channels,
                    num_heads_upsample=num_heads_upsample, use_scale_shift_norm=use_scale_shift_norm,
                    resblock_updown=resblock_updown, use_new_attention_order=use_new_attention_order,
                    use_spatial_transformer=use_spatial_transformer, transformer_depth=transformer_depth,
                    context_dim=context_dim, n_embed=n_embed, legacy=legacy,
                    # disable_self_attentions=disable_self_attentions,
                    # num_attention_blocks=num_attention_blocks,
                    # disable_middle_self_attn=disable_middle_self_attn,
                    use_linear_in_transformer=use_linear_in_transformer,
                    infusion2control=infusion2control,
                    guiding=guiding, two_stream_mode=two_stream_mode, control_model_ratio=control_model_ratio, fixed=fixed,
                    )  # initialise pretrained model
            else:
                self.control_model = ControlledUNetModel(
                    image_size=image_size, in_channels=in_channels, model_channels=model_channels,
                    out_channels=out_channels, num_res_blocks=num_res_blocks,
                    attention_resolutions=attention_resolutions, dropout=dropout, channel_mult=channel_mult,
                    conv_resample=conv_resample, dims=dims, use_checkpoint=use_checkpoint,
                    use_fp16=use_fp16, num_heads=num_heads, num_head_channels=num_head_channels,
                    num_heads_upsample=num_heads_upsample, use_scale_shift_norm=use_scale_shift_norm,
                    resblock_updown=resblock_updown, use_new_attention_order=use_new_attention_order,
                    use_spatial_transformer=use_spatial_transformer, transformer_depth=transformer_depth,
                    context_dim=context_dim, n_embed=n_embed, legacy=legacy,
                    # disable_self_attentions=disable_self_attentions,
                    # num_attention_blocks=num_attention_blocks,
                    # disable_middle_self_attn=disable_middle_self_attn,
                    use_linear_in_transformer=use_linear_in_transformer,
                    infusion2control=infusion2control,
                    guiding=guiding, two_stream_mode=two_stream_mode, control_model_ratio=control_model_ratio,
                    )  # initialise pretrained model

            if not learn_embedding:
                del self.control_model.time_embed
                # del self.control_model.label_emb

            # if guiding in ('encoder', 'encoder_double'):
            #     self.control_model.output_blocks = None
            ################# end control model variations #################

            self.enc_zero_convs_out = nn.ModuleList([])
            self.enc_zero_convs_in = nn.ModuleList([])

            self.middle_block_out = None
            self.middle_block_in = None

            self.dec_zero_convs_out = nn.ModuleList([])
            self.dec_zero_convs_in = nn.ModuleList([])

            ch_inout_ctr = {'enc': [], 'mid': [], 'dec': []}
            ch_inout_base = {'enc': [], 'mid': [], 'dec': []}

            ################# Gather Channel Sizes #################
            for module in self.control_model.input_blocks:
                if isinstance(module[0], nn.Conv2d):
                    ch_inout_ctr['enc'].append((module[0].in_channels, module[0].out_channels))
                elif isinstance(module[0], (ResBlock, ResBlock_orig)):
                    ch_inout_ctr['enc'].append((module[0].channels, module[0].out_channels))
                elif isinstance(module[0], Downsample):
                    ch_inout_ctr['enc'].append((module[0].channels, module[-1].out_channels))

            for module in base_model.input_blocks:
                if isinstance(module[0], nn.Conv2d):
                    ch_inout_base['enc'].append((module[0].in_channels, module[0].out_channels))
                elif isinstance(module[0], (ResBlock, ResBlock_orig)):
                    ch_inout_base['enc'].append((module[0].channels, module[0].out_channels))
                elif isinstance(module[0], Downsample):
                    ch_inout_base['enc'].append((module[0].channels, module[-1].out_channels))

            ch_inout_ctr['mid'].append((self.control_model.middle_block[0].channels, self.control_model.middle_block[-1].out_channels))
            ch_inout_base['mid'].append((base_model.middle_block[0].channels, base_model.middle_block[-1].out_channels))

            if guiding not in ('encoder', 'encoder_double'):
                for module in self.control_model.output_blocks:
                    if isinstance(module[0], nn.Conv2d):
                        ch_inout_ctr['dec'].append((module[0].in_channels, module[0].out_channels))
                    elif isinstance(module[0], (ResBlock, ResBlock_orig)):
                        ch_inout_ctr['dec'].append((module[0].channels, module[0].out_channels))
                    elif isinstance(module[-1], Upsample):
                        ch_inout_ctr['dec'].append((module[0].channels, module[-1].out_channels))

            for module in base_model.output_blocks:
                if isinstance(module[0], nn.Conv2d):
                    ch_inout_base['dec'].append((module[0].in_channels, module[0].out_channels))
                elif isinstance(module[0], (ResBlock, ResBlock_orig)):
                    ch_inout_base['dec'].append((module[0].channels, module[0].out_channels))
                elif isinstance(module[-1], Upsample):
                    ch_inout_base['dec'].append((module[0].channels, module[-1].out_channels))

            self.ch_inout_ctr = ch_inout_ctr
            self.ch_inout_base = ch_inout_base

            ################# Build zero convolutions #################
            if two_stream_mode == 'cross':
                ################# cross infusion #################
                # infusion2control
                # add
                if infusion2control == 'add':
                    for i in range(len(ch_inout_base['enc'])):
                        self.enc_zero_convs_in.append(self.make_zero_conv(
                            in_channels=ch_inout_base['enc'][i][1], out_channels=ch_inout_ctr['enc'][i][1])
                            )

                    if guiding == 'full':
                        self.middle_block_in = self.make_zero_conv(ch_inout_base['mid'][-1][1], ch_inout_ctr['mid'][-1][1])
                        for i in range(len(ch_inout_base['dec']) - 1):
                            self.dec_zero_convs_in.append(self.make_zero_conv(
                                in_channels=ch_inout_base['dec'][i][1], out_channels=ch_inout_ctr['dec'][i][1])
                                )

                    # cat - processing full concatenation (all output layers are concatenated without "slimming")
                elif infusion2control == 'cat':
                    for ch_io_base in ch_inout_base['enc']:
                        self.enc_zero_convs_in.append(self.make_zero_conv(
                            in_channels=ch_io_base[1], out_channels=ch_io_base[1])
                            )

                    if guiding == 'full':
                        self.middle_block_in = self.make_zero_conv(ch_inout_base['mid'][-1][1], ch_inout_base['mid'][-1][1])
                        for ch_io_base in ch_inout_base['dec']:
                            self.dec_zero_convs_in.append(self.make_zero_conv(
                                in_channels=ch_io_base[1], out_channels=ch_io_base[1])
                                )

                elif infusion2control == 'att':
                    for i in range(len(ch_inout_base['enc'])):
                        self.enc_zero_convs_in.append(CrossAttInfusion(
                            in_channels=ch_inout_ctr['enc'][i][1], context_dim=ch_inout_base['enc'][i][1],
                            n_heads=n_heads, d_head=d_head
                        ))

                    if guiding == 'full':
                        self.middle_block_in = CrossAttInfusion(
                            in_channels=ch_inout_ctr['mid'][-1][1], context_dim=ch_inout_base['mid'][-1][1],
                            n_heads=n_heads, d_head=d_head
                        )
                        for i in range(len(ch_inout_base['dec']) - 1):
                            self.dec_zero_convs_in.append(CrossAttInfusion(
                                in_channels=ch_inout_ctr['dec'][i][1], context_dim=ch_inout_base['dec'][i][1],
                                n_heads=n_heads, d_head=d_head
                            ))

                    # None - no changes

                # infusion2base - consider all three guidings
                    # add
                if infusion2base == 'add':
                    self.middle_block_out = self.make_zero_conv(ch_inout_ctr['mid'][-1][1], ch_inout_base['mid'][-1][1])

                    if guiding in ('encoder', 'encoder_double'):
                        self.dec_zero_convs_out.append(
                                self.make_zero_conv(ch_inout_ctr['enc'][-1][1], ch_inout_base['mid'][-1][1])
                                )
                        for i in range(1, len(ch_inout_ctr['enc'])):
                            self.dec_zero_convs_out.append(
                                self.make_zero_conv(ch_inout_ctr['enc'][-(i+1)][1], ch_inout_base['dec'][i-1][1])
                                )
                    if guiding in ('encoder_double', 'full'):
                        for i in range(len(ch_inout_ctr['enc'])):
                            self.enc_zero_convs_out.append(self.make_zero_conv(
                                in_channels=ch_inout_ctr['enc'][i][1], out_channels=ch_inout_base['enc'][i][1])
                                )

                    if guiding == 'full':
                        for i in range(len(ch_inout_ctr['dec'])):
                            self.dec_zero_convs_out.append(self.make_zero_conv(
                                in_channels=ch_inout_ctr['dec'][i][1], out_channels=ch_inout_base['dec'][i][1])
                                )
                            
                elif self.infusion2base == 'att':
                    self.middle_block_out = CrossAttInfusion(
                        in_channels=ch_inout_base['mid'][-1][1], context_dim=ch_inout_ctr['mid'][-1][1],
                        n_heads=n_heads, d_head=d_head,
                    )

                    if self.guiding in ('encoder', 'encoder_double'):
                        self.dec_zero_convs_out.append(CrossAttInfusion(
                            in_channels=ch_inout_base['mid'][-1][1], context_dim=ch_inout_ctr['enc'][-1][1],
                            n_heads=n_heads, d_head=d_head,
                        ))

                        for i in range(1, len(ch_inout_ctr['enc'])):
                            self.dec_zero_convs_out.append(CrossAttInfusion(
                                in_channels=ch_inout_base['dec'][i-1][1], context_dim=ch_inout_ctr['enc'][-(i+1)][1],
                                n_heads=n_heads, d_head=d_head,
                            ))

                    if self.guiding in ('encoder_double', 'full'):
                        for i in range(len(ch_inout_ctr['enc'])):
                            self.enc_zero_convs_out.append(CrossAttInfusion(
                                in_channels=ch_inout_base['enc'][i][1], context_dim=ch_inout_ctr['enc'][i][1],
                                n_heads=n_heads, d_head=d_head,
                            ))

                    if self.guiding == 'full':
                        for i in range(len(ch_inout_ctr['dec'])):
                            self.dec_zero_convs_out.append(CrossAttInfusion(
                                in_channels=ch_inout_base['dec'][i][1], context_dim=ch_inout_ctr['dec'][i][1],
                                n_heads=n_heads, d_head=d_head,
                            ))

                    # cat
                        # TODO after everything

            elif two_stream_mode == 'sequential':
                ################# sequential infusion #################
                # infusion2control (only difference to before is the skipping of down/upsampling zero convolutions)
                # add
                if infusion2control == 'add':
                    for i in range(len(ch_inout_base['enc'])):
                        if isinstance(base_model.input_blocks[min(i + 1, len(ch_inout_base['enc']) - 1)][-1], Downsample):
                            self.enc_zero_convs_in.append(nn.Identity())
                            continue
                        self.enc_zero_convs_in.append(self.make_zero_conv(
                            in_channels=ch_inout_base['enc'][i][1], out_channels=ch_inout_ctr['enc'][i][1])
                            )

                    if guiding == 'full':
                        self.middle_block_in = self.make_zero_conv(ch_inout_base['mid'][-1][1], ch_inout_ctr['mid'][-1][1])
                        for i in range(len(ch_inout_base['dec']) - 1):
                            if isinstance(base_model.output_blocks[min(i + 1, len(ch_inout_base['dec']) - 1)][-1], Upsample):
                                self.enc_zero_convs_in.append(nn.Identity())
                                self.dec_zero_convs_in.append(nn.Identity())
                                continue
                            self.dec_zero_convs_in.append(self.make_zero_conv(
                                in_channels=ch_inout_base['dec'][i][1], out_channels=ch_inout_ctr['dec'][i][1])
                                )
                        self.dec_zero_convs_in.append(self.make_zero_conv(
                                in_channels=ch_inout_base['dec'][-1][1], out_channels=ch_inout_ctr['dec'][-1][1])
                                )

                    # cat - processing full concatenation (all output layers are concatenated without "slimming")
                if infusion2control == 'cat':
                    for i in range(len(ch_inout_base['enc'])):
                        if isinstance(base_model.input_blocks[min(i + 1, len(ch_inout_base['enc']) - 1)][-1], Downsample):
                            self.enc_zero_convs_in.append(nn.Identity())
                            continue
                        self.enc_zero_convs_in.append(self.make_zero_conv(
                            in_channels=ch_inout_base['enc'][i][1], out_channels=ch_inout_base['enc'][i][1])
                            )

                    if guiding == 'full':
                        self.middle_block_in = self.make_zero_conv(ch_inout_base['mid'][-1][1], ch_inout_ctr['mid'][-1][1])
                        for i in range(len(ch_inout_base['dec']) - 1):
                            if isinstance(base_model.output_blocks[min(i + 1, len(ch_inout_base['dec']) - 1)][-1], Upsample):
                                self.enc_zero_convs_in.append(nn.Identity())
                                self.dec_zero_convs_in.append(nn.Identity())
                                continue
                            self.dec_zero_convs_in.append(self.make_zero_conv(
                                in_channels=ch_inout_base['dec'][i][1], out_channels=ch_inout_base['dec'][i][1])
                                )
                        self.dec_zero_convs_in.append(self.make_zero_conv(
                                in_channels=ch_inout_base['dec'][-1][1], out_channels=ch_inout_ctr['dec'][-1][1])
                                )

                    # None - no changes

                # infusion2base
                    # add
                if infusion2base == 'add':
                    self.middle_block_out = self.make_zero_conv(ch_inout_ctr['mid'][-1][1], ch_inout_base['enc'][-1][1])

                    if guiding in ('encoder', 'encoder_double'):
                        self.dec_zero_convs_out.append(
                            self.make_zero_conv(ch_inout_ctr['enc'][-1][1], ch_inout_base['mid'][-1][1])
                            )
                        for i in range(len(ch_inout_ctr['enc']) - 1):
                            # if isinstance(self.ch_inout_ctr[i], Upsample): continue
                            if isinstance(base_model.output_blocks[i+1][-1], Upsample):
                                continue
                            self.dec_zero_convs_out.append(self.make_zero_conv(ch_inout_ctr['enc'][::-1][i+1][1], ch_inout_base['dec'][i][1]))

                    if guiding in ('encoder_double', 'full'):
                        self.enc_zero_convs_out.append(self.make_zero_conv(
                                in_channels=ch_inout_ctr['enc'][0][1], out_channels=ch_inout_base['enc'][0][0])
                                )
                        for i in range(1, len(ch_inout_ctr['enc'])):
                            if isinstance(base_model.input_blocks[i][0], Downsample):
                                continue
                            self.enc_zero_convs_out.append(self.make_zero_conv(
                                in_channels=ch_inout_ctr['enc'][i][1], out_channels=ch_inout_base['enc'][i - 1][1])
                                )

                    if guiding == 'full':
                        self.dec_zero_convs_out.append(
                            self.make_zero_conv(ch_inout_ctr['dec'][0][1], ch_inout_base['mid'][-1][1])
                            )
                        for i in range(len(ch_inout_ctr['dec']) - 1):
                            # if isinstance(self.control_model.output_blocks[i][-1], Upsample): continue
                            if isinstance(base_model.output_blocks[i+1][-1], Upsample):
                                continue
                            self.dec_zero_convs_out.append(self.make_zero_conv(
                                in_channels=ch_inout_ctr['dec'][i+1][1], out_channels=ch_inout_base['dec'][i][1])
                                )

                    # cat
                        # TODO after everything

        self.input_hint_block = TimestepEmbedSequential(
            conv_nd(dims, hint_channels, 16, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 16, 16, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 16, 32, 3, padding=1, stride=2),
            nn.SiLU(),
            conv_nd(dims, 32, 32, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 32, 96, 3, padding=1, stride=2),
            nn.SiLU(),
            conv_nd(dims, 96, 96, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 96, 256, 3, padding=1, stride=2),
            nn.SiLU(),
            zero_module(conv_nd(dims, 256, max(1, int(model_channels * control_model_ratio)), 3, padding=1))
        )

        scale_list = [1.] * len(self.enc_zero_convs_out) + [1.] + [1.] * len(self.dec_zero_convs_out)
        self.register_buffer('scale_list', torch.tensor(scale_list))

    def make_zero_conv(self, in_channels, out_channels=None):
        in_channels = in_channels
        out_channels = out_channels or in_channels
        return TimestepEmbedSequential(
            zero_module(conv_nd(self.dims, in_channels, out_channels, 1, padding=0))
            )

    def delete_this(self, n_heads, d_head):
        self.enc_zero_convs_out = nn.ModuleList([])
        self.enc_zero_convs_in = nn.ModuleList([])

        self.middle_block_out = None
        self.middle_block_in = None

        self.dec_zero_convs_out = nn.ModuleList([])
        self.dec_zero_convs_in = nn.ModuleList([])

        ch_inout_ctr = {'enc': [], 'mid': [], 'dec': []}
        ch_inout_base = {'enc': [], 'mid': [], 'dec': []}

        ################# Gather Channel Sizes #################
        for module in self.control_model.input_blocks:
            if isinstance(module[0], nn.Conv2d):
                ch_inout_ctr['enc'].append((module[0].in_channels, module[0].out_channels))
            elif isinstance(module[0], (ResBlock, ResBlock_orig)):
                ch_inout_ctr['enc'].append((module[0].channels, module[0].out_channels))
            elif isinstance(module[0], Downsample):
                ch_inout_ctr['enc'].append((module[0].channels, module[-1].out_channels))

        for module in self.base_model.input_blocks:
            if isinstance(module[0], nn.Conv2d):
                ch_inout_base['enc'].append((module[0].in_channels, module[0].out_channels))
            elif isinstance(module[0], (ResBlock, ResBlock_orig)):
                ch_inout_base['enc'].append((module[0].channels, module[0].out_channels))
            elif isinstance(module[0], Downsample):
                ch_inout_base['enc'].append((module[0].channels, module[-1].out_channels))

        ch_inout_ctr['mid'].append((self.control_model.middle_block[0].channels, self.control_model.middle_block[-1].out_channels))
        ch_inout_base['mid'].append((self.base_model.middle_block[0].channels, self.base_model.middle_block[-1].out_channels))

        if self.guiding not in ('encoder', 'encoder_double'):
            for module in self.control_model.output_blocks:
                if isinstance(module[0], nn.Conv2d):
                    ch_inout_ctr['dec'].append((module[0].in_channels, module[0].out_channels))
                elif isinstance(module[0], (ResBlock, ResBlock_orig)):
                    ch_inout_ctr['dec'].append((module[0].channels, module[0].out_channels))
                elif isinstance(module[-1], Upsample):
                    ch_inout_ctr['dec'].append((module[0].channels, module[-1].out_channels))

        for module in self.base_model.output_blocks:
            if isinstance(module[0], nn.Conv2d):
                ch_inout_base['dec'].append((module[0].in_channels, module[0].out_channels))
            elif isinstance(module[0], (ResBlock, ResBlock_orig)):
                ch_inout_base['dec'].append((module[0].channels, module[0].out_channels))
            elif isinstance(module[-1], Upsample):
                ch_inout_base['dec'].append((module[0].channels, module[-1].out_channels))

        self.ch_inout_ctr = ch_inout_ctr
        self.ch_inout_base = ch_inout_base

        ################# Build zero convolutions #################

        if self.two_stream_mode == 'cross':
            ################# cross infusion #################
            # infusion2control
            # add
            if self.infusion2control == 'add':
                for i in range(len(ch_inout_base['enc'])):
                    self.enc_zero_convs_in.append(self.make_zero_conv(
                        in_channels=ch_inout_base['enc'][i][1], out_channels=ch_inout_ctr['enc'][i][1])
                        )

                if self.guiding == 'full':
                    self.middle_block_in = self.make_zero_conv(ch_inout_base['mid'][-1][1], ch_inout_ctr['mid'][-1][1])
                    for i in range(len(ch_inout_base['dec']) - 1):
                        self.dec_zero_convs_in.append(self.make_zero_conv(
                            in_channels=ch_inout_base['dec'][i][1], out_channels=ch_inout_ctr['dec'][i][1])
                            )

                # cat - processing full concatenation (all output layers are concatenated without "slimming")
            elif self.infusion2control == 'cat':
                for ch_io_base in ch_inout_base['enc']:
                    self.enc_zero_convs_in.append(self.make_zero_conv(
                        in_channels=ch_io_base[1], out_channels=ch_io_base[1])
                        )

                if self.guiding == 'full':
                    self.middle_block_in = self.make_zero_conv(ch_inout_base['mid'][-1][1], ch_inout_base['mid'][-1][1])
                    for ch_io_base in ch_inout_base['dec']:
                        self.dec_zero_convs_in.append(self.make_zero_conv(
                            in_channels=ch_io_base[1], out_channels=ch_io_base[1])
                            )

            elif self.infusion2control == 'att':

                for i in range(len(ch_inout_base['enc'])):
                    self.enc_zero_convs_in.append(CrossAttInfusion(
                        in_channels=ch_inout_ctr['enc'][i][1], context_dim=ch_inout_base['enc'][i][1],
                        n_heads=n_heads, d_head=d_head
                    ))

                if self.guiding == 'full':
                    self.middle_block_in = CrossAttInfusion(
                        in_channels=ch_inout_ctr['mid'][-1][1], context_dim=ch_inout_base['mid'][-1][1],
                        n_heads=n_heads, d_head=d_head
                    )
                    for i in range(len(ch_inout_base['dec']) - 1):
                        self.dec_zero_convs_in.append(CrossAttInfusion(
                            in_channels=ch_inout_ctr['dec'][i][1], context_dim=ch_inout_base['dec'][i][1],
                            n_heads=n_heads, d_head=d_head
                        ))

                # None - no changes

            # infusion2base - consider all three guidings
                # add
            if self.infusion2base == 'add':
                self.middle_block_out = self.make_zero_conv(ch_inout_ctr['mid'][-1][1], ch_inout_base['mid'][-1][1])

                if self.guiding in ('encoder', 'encoder_double'):
                    self.dec_zero_convs_out.append(
                            self.make_zero_conv(ch_inout_ctr['enc'][-1][1], ch_inout_base['mid'][-1][1])
                            )
                    for i in range(1, len(ch_inout_ctr['enc'])):
                        self.dec_zero_convs_out.append(
                            self.make_zero_conv(ch_inout_ctr['enc'][-(i+1)][1], ch_inout_base['dec'][i-1][1])
                            )
                if self.guiding in ('encoder_double', 'full'):
                    for i in range(len(ch_inout_ctr['enc'])):
                        self.enc_zero_convs_out.append(self.make_zero_conv(
                            in_channels=ch_inout_ctr['enc'][i][1], out_channels=ch_inout_base['enc'][i][1])
                            )

                if self.guiding == 'full':
                    for i in range(len(ch_inout_ctr['dec'])):
                        self.dec_zero_convs_out.append(self.make_zero_conv(
                            in_channels=ch_inout_ctr['dec'][i][1], out_channels=ch_inout_base['dec'][i][1])
                            )

            elif self.infusion2base == 'att':
                self.middle_block_out = CrossAttInfusion(
                    in_channels=ch_inout_base['mid'][-1][1], context_dim=ch_inout_ctr['mid'][-1][1],
                    n_heads=n_heads, d_head=d_head,
                )

                if self.guiding in ('encoder', 'encoder_double'):
                    self.dec_zero_convs_out.append(CrossAttInfusion(
                        in_channels=ch_inout_base['mid'][-1][1], context_dim=ch_inout_ctr['enc'][-1][1],
                        n_heads=n_heads, d_head=d_head,
                    ))

                    for i in range(1, len(ch_inout_ctr['enc'])):
                        self.dec_zero_convs_out.append(CrossAttInfusion(
                            in_channels=ch_inout_base['dec'][i-1][1], context_dim=ch_inout_ctr['enc'][-(i+1)][1],
                            n_heads=n_heads, d_head=d_head,
                        ))

                if self.guiding in ('encoder_double', 'full'):
                    for i in range(len(ch_inout_ctr['enc'])):
                        self.enc_zero_convs_out.append(CrossAttInfusion(
                            in_channels=ch_inout_base['enc'][i][1], context_dim=ch_inout_ctr['enc'][i][1],
                            n_heads=n_heads, d_head=d_head,
                        ))

                if self.guiding == 'full':
                    for i in range(len(ch_inout_ctr['dec'])):
                        self.dec_zero_convs_out.append(CrossAttInfusion(
                            in_channels=ch_inout_base['dec'][i][1], context_dim=ch_inout_ctr['dec'][i][1],
                            n_heads=n_heads, d_head=d_head,
                        ))

            self.ch_inout_ctr = ch_inout_ctr
            self.ch_inout_base = ch_inout_base

    def infuse(self, stream, infusion, mlp, variant, emb, scale=1.0):
        if variant == 'add':
            stream = stream + mlp(infusion, emb) * scale
        elif variant == 'cat':
            stream = torch.cat([stream, mlp(infusion, emb)], dim=1)
        elif variant == 'att':
            stream = mlp(stream, infusion)

        return stream

    def scale_controls(self, scale):

        possible_blocks = [
                    self.enc_zero_convs_out,
                    [self.middle_block_out],
                    self.dec_zero_convs_out
                    ]
        for block in possible_blocks:
            scale_list = [scale ** n for n in range(len(block))]
            for subblock, factor in zip(block, scale_list):
                for conv in subblock:
                    conv.weight = torch.nn.Parameter(conv.weight * factor)
                    conv.bias = torch.nn.Parameter(conv.bias * factor)

    def forward(self, x, hint, timesteps, context, base_model, precomputed_hint=False, **kwargs):
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)

        if self.learn_embedding:
            emb = self.control_model.time_embed(t_emb)
        else:
            emb = base_model.time_embed(t_emb)

        if precomputed_hint:
            guided_hint = hint
        else:
            guided_hint = self.input_hint_block(hint, emb, context)

        h_ctr = h_base = x.type(base_model.dtype)
        hs_base = []
        hs_ctr = []
        it_enc_convs_in = iter(self.enc_zero_convs_in)
        it_enc_convs_out = iter(self.enc_zero_convs_out)
        it_dec_convs_in = iter(self.dec_zero_convs_in)
        it_dec_convs_out = iter(self.dec_zero_convs_out)

        ###################### Cross Control        ######################

        if self.two_stream_mode == 'cross':
            # input blocks (encoder)
            for module_base, module_ctr in zip(base_model.input_blocks, self.control_model.input_blocks):
                h_base = module_base(h_base, emb, context)
                h_ctr = module_ctr(h_ctr, emb, context)
                if guided_hint is not None:
                    h_ctr = h_ctr + guided_hint
                    # h_ctr = torch.cat([h_ctr, guided_hint], dim=1)
                    guided_hint = None

                if not self.debugger['skip_enc_infusion']:
                    h_base = self.infuse(h_base, h_ctr, next(it_enc_convs_out), self.infusion2base, emb)
                    # if self.guiding in ('encoder_double', 'full'):
                    #     if self.infusion2base == 'add':
                    #         h_base = h_base + next(it_enc_convs_out)(h_ctr, emb)
                    #     elif self.infusion2base == 'cat':
                    #         raise NotImplementedError()

                hs_base.append(h_base)
                hs_ctr.append(h_ctr)

                h_ctr = self.infuse(h_ctr, h_base, next(it_enc_convs_in), self.infusion2control, emb)
                # if self.infusion2control == 'add':
                #     h_ctr = h_ctr + next(it_enc_convs_in)(h_base, emb)
                # elif self.infusion2control == 'cat':
                #     h_ctr = th.cat([h_ctr, next(it_enc_convs_in)(h_base, emb)], dim=1)

            # mid blocks (bottleneck)
            h_base = base_model.middle_block(h_base, emb, context)
            h_ctr = self.control_model.middle_block(h_ctr, emb, context)

            h_base = self.infuse(h_base, h_ctr, self.middle_block_out, self.infusion2base, emb)
            # if self.infusion2base == 'add':
            #     h_base = h_base + self.middle_block_out(h_ctr, emb)
            # elif self.infusion2base == 'cat':
            #     raise NotImplementedError()

            if self.guiding == 'full':
                h_ctr = self.infuse(h_ctr, h_base, self.middle_block_in, self.infusion2control, emb)
                # if self.infusion2control == 'add':
                #     h_ctr = h_ctr + self.middle_block_in(h_base, emb)
                # elif self.infusion2control == 'cat':
                #     h_ctr = th.cat([h_ctr, self.middle_block_in(h_base, emb)], dim=1)

            # output blocks (decoder)
            for module_base, module_ctr in zip(
                    base_model.output_blocks,
                    self.control_model.output_blocks if hasattr(
                    self.control_model, 'output_blocks') else [None] * len(base_model.output_blocks)
                    ):

                if not self.debugger['skip_dec_infusion']:
                    if self.guiding != 'full':

                        h_base = self.infuse(h_base, hs_ctr.pop(), next(it_dec_convs_out), self.infusion2base, emb)
                        # if self.infusion2base == 'add':
                        #     h_base = h_base + next(it_dec_convs_out)(hs_ctr.pop(), emb)
                        # elif self.infusion2base == 'cat':
                        #     raise NotImplementedError()

                h_base = th.cat([h_base, hs_base.pop()], dim=1)
                h_base = module_base(h_base, emb, context)
                # if self.guiding == 'full':
                #     h_ctr = th.cat([h_ctr, hs_ctr.pop()], dim=1)
                #     h_ctr = module_ctr(h_ctr, emb, context)
                #     if self.infusion2base == 'add':
                #         h_base = h_base + next(it_dec_convs_out)(h_ctr, emb)
                #     elif self.infusion2base == 'cat':
                #         raise NotImplementedError()

                #     if module_base != base_model.output_blocks[-1]:
                #         if self.infusion2control == 'add':
                #             h_ctr = h_ctr + next(it_dec_convs_in)(h_base, emb)
                #         elif self.infusion2control == 'cat':
                #             h_ctr = th.cat([h_ctr, next(it_dec_convs_in)(h_base, emb)], dim=1)

                ##### Quick and dirty attempt of fixing "full" with not applying corrections to the last layer #####
                if not self.debugger['skip_dec_infusion']:
                    if self.guiding == 'full':
                        h_ctr = th.cat([h_ctr, hs_ctr.pop()], dim=1)
                        h_ctr = module_ctr(h_ctr, emb, context)
                        if module_base != base_model.output_blocks[-1]:
                            h_base = self.infuse(h_base, h_ctr, next(it_dec_convs_out), self.infusion2base, emb)
                            h_ctr = self.infuse(h_ctr, h_base, next(it_dec_convs_in), self.infusion2control, emb)
                            # if self.infusion2base == 'add':
                            #     h_base = h_base + next(it_dec_convs_out)(h_ctr, emb)
                            # elif self.infusion2base == 'cat':
                            #     raise NotImplementedError()

                            # if self.infusion2control == 'add':
                            #     h_ctr = h_ctr + next(it_dec_convs_in)(h_base, emb)
                            # elif self.infusion2control == 'cat':
                            #     h_ctr = th.cat([h_ctr, next(it_dec_convs_in)(h_base, emb)], dim=1)

        ###################### Sequential Control   ######################
        elif self.two_stream_mode == 'sequential':
            # input blocks (encoder)
            for module_base, module_ctr in zip(base_model.input_blocks, self.control_model.input_blocks):
                h_ctr = module_ctr(h_ctr, emb, context)
                if guided_hint is not None:
                    h_ctr = h_ctr + guided_hint
                    # h_ctr = torch.cat([h_ctr, guided_hint], dim=1)
                    guided_hint = None

                if not self.debugger['skip_enc_infusion']:
                    if self.guiding in ('encoder_double', 'full') and not isinstance(module_ctr[0], Downsample):
                        if self.infusion2base == 'add':
                            next_module = next(it_enc_convs_out)
                            h_base = h_base + next_module(h_ctr, emb)
                        elif self.infusion2base == 'cat':
                            raise NotImplementedError()

                h_base = module_base(h_base, emb, context)
                hs_base.append(h_base)
                hs_ctr.append(h_ctr)

                if self.infusion2control is not None:
                    infusion_conv = next(it_enc_convs_in)
                    if not isinstance(infusion_conv, nn.Identity):
                        if self.infusion2control == 'add':
                            h_ctr = h_ctr + infusion_conv(h_base, emb)
                        elif self.infusion2control == 'cat':
                            h_ctr = th.cat([h_ctr, infusion_conv(h_base, emb)], dim=1)

            # mid blocks (bottleneck)
            h_ctr = self.control_model.middle_block(h_ctr, emb, context)

            if self.infusion2base == 'add':
                h_base = h_base + self.middle_block_out(h_ctr, emb)
            elif self.infusion2base == 'cat':
                raise NotImplementedError()

            h_base = base_model.middle_block(h_base, emb, context)

            if self.guiding == 'full':
                if self.infusion2control == 'add':
                    h_ctr = h_ctr + self.middle_block_in(h_base, emb)
                elif self.infusion2control == 'cat':
                    h_ctr = th.cat([h_ctr, self.middle_block_in(h_base, emb)], dim=1)

            # output blocks (decoder)
            for module_base, module_ctr in zip(
                    base_model.output_blocks,
                    self.control_model.output_blocks if hasattr(
                        self.control_model, 'output_blocks') else [None] * len(base_model.output_blocks)
                    ):

                if not self.debugger['skip_dec_infusion']:
                    if self.guiding == 'full':
                        h_ctr = th.cat([h_ctr, hs_ctr.pop()], dim=1)
                        h_ctr = module_ctr(h_ctr, emb, context)

                        if not isinstance(module_ctr[-1], Upsample):
                            if self.infusion2base == 'add':
                                h_base = h_base + next(it_dec_convs_out)(h_ctr, emb)
                            elif self.infusion2base == 'cat':
                                raise NotImplementedError()
                    else:
                        if isinstance(module_base[-1], Upsample):
                            hs_ctr.pop()
                        else:
                            if self.infusion2base == 'add':
                                h_base = h_base + next(it_dec_convs_out)(hs_ctr.pop(), emb)
                            elif self.infusion2base == 'cat':
                                raise NotImplementedError()

                h_base = th.cat([h_base, hs_base.pop()], dim=1)
                h_base = module_base(h_base, emb, context)

                if not self.debugger['skip_dec_infusion']:
                    if self.infusion2control is not None and self.guiding == 'full':
                        infusion_conv = next(it_dec_convs_in)
                        if not isinstance(infusion_conv, nn.Identity):
                            if self.infusion2control == 'add':
                                h_ctr = h_ctr + infusion_conv(h_base, emb)
                            elif self.infusion2control == 'cat':
                                h_ctr = th.cat([h_ctr, infusion_conv(h_base, emb)], dim=1)

        return base_model.out(h_base)


class ControlledUNetModel(nn.Module):
    """
    The full UNet model with attention and timestep embedding.
    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param num_res_blocks: number of residual blocks per downsample.
    :param attention_resolutions: a collection of downsample rates at which
        attention will take place. May be a set, list, or tuple.
        For example, if this contains 4, then at 4x downsampling, attention
        will be used.
    :param dropout: the dropout probability.
    :param channel_mult: channel multiplier for each level of the UNet.
    :param conv_resample: if True, use learned convolutions for upsampling and
        downsampling.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param num_classes: if specified (as an int), then this model will be
        class-conditional with `num_classes` classes.
    :param use_checkpoint: use gradient checkpointing to reduce memory usage.
    :param num_heads: the number of attention heads in each attention layer.
    :param num_heads_channels: if specified, ignore num_heads and instead use
                               a fixed channel width per attention head.
    :param num_heads_upsample: works with num_heads to set a different number
                               of heads for upsampling. Deprecated.
    :param use_scale_shift_norm: use a FiLM-like conditioning mechanism.
    :param resblock_updown: use residual blocks for up/downsampling.
    :param use_new_attention_order: use a different attention pattern for potentially
                                    increased efficiency.
    """

    def __init__(
        self,
        image_size,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        num_classes=None,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=-1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_new_attention_order=False,
        use_spatial_transformer=False,    # custom transformer support
        transformer_depth=1,              # custom transformer support
        context_dim=None,                 # custom transformer support
        n_embed=None,                     # custom support for prediction of discrete ids into codebook of first stage vq model
        legacy=True,
        disable_self_attentions=None,
        num_attention_blocks=None,
        disable_middle_self_attn=False,
        use_linear_in_transformer=False,
        infusion2control='cat',         # how to infuse intermediate information into the control net? {'add', 'cat', None}
        guiding='encoder',              # use just encoder for control or the whole encoder + decoder net? {'encoder', 'encoder_double', 'full'}
        two_stream_mode='cross',        # mode for the two stream infusion. {'cross', 'sequential'}
        control_model_ratio=1.0,
    ):
        super().__init__()
        if use_spatial_transformer:
            assert context_dim is not None, 'Fool!! You forgot to include the dimension of your cross-attention conditioning...'

        if context_dim is not None:
            assert use_spatial_transformer, 'Fool!! You forgot to use the spatial transformer for your cross-attention conditioning...'
            from omegaconf.listconfig import ListConfig
            if type(context_dim) == ListConfig:
                context_dim = list(context_dim)

        self.infusion2control = infusion2control
        # infusion_factor = 2 if infusion2control == 'cat' else 1
        infusion_factor = int(1 / control_model_ratio)
        cat_infusion = 1 if infusion2control == 'cat' else 0

        self.guiding = guiding
        self.two_stage_mode = two_stream_mode
        # seq_factor = 1 if two_stream_mode == 'sequential' and infusion2control == 'cat' else 0
        seq_factor = 1 if two_stream_mode == 'sequential' and infusion2control == 'cat' else 0

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        if num_heads == -1:
            assert num_head_channels != -1, 'Either num_heads or num_head_channels has to be set'

        if num_head_channels == -1:
            assert num_heads != -1, 'Either num_heads or num_head_channels has to be set'

        self.image_size = image_size
        self.in_channels = in_channels
        # self.model_channels = model_channels
        self.out_channels = out_channels
        if isinstance(num_res_blocks, int):
            self.num_res_blocks = len(channel_mult) * [num_res_blocks]
        else:
            if len(num_res_blocks) != len(channel_mult):
                raise ValueError("provide num_res_blocks either as an int (globally constant) or "
                                 "as a list/tuple (per-level) with the same length as channel_mult")
            self.num_res_blocks = num_res_blocks
        if disable_self_attentions is not None:
            # should be a list of booleans, indicating whether to disable self-attention in TransformerBlocks or not
            assert len(disable_self_attentions) == len(channel_mult)
        if num_attention_blocks is not None:
            assert len(num_attention_blocks) == len(self.num_res_blocks)
            assert all(map(lambda i: self.num_res_blocks[i] >= num_attention_blocks[i], range(len(num_attention_blocks))))
            print(f"Constructor of UNetModel received num_attention_blocks={num_attention_blocks}. "
                  f"This option has LESS priority than attention_resolutions {attention_resolutions}, "
                  f"i.e., in cases where num_attention_blocks[i] > 0 but 2**i not in attention_resolutions, "
                  f"attention will still not be set.")

        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        self.predict_codebook_ids = n_embed is not None

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        model_channels = max(1, int(model_channels * control_model_ratio))
        self.model_channels = model_channels
        self.control_model_ratio = control_model_ratio

        if self.num_classes is not None:
            if isinstance(self.num_classes, int):
                self.label_emb = nn.Embedding(num_classes, time_embed_dim)
            elif self.num_classes == "continuous":
                print("setting up linear c_adm embedding layer")
                self.label_emb = nn.Linear(1, time_embed_dim)
            else:
                raise ValueError()

        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential(
                    conv_nd(dims, in_channels, model_channels, 3, padding=1)
                )
            ]
        )
        self._feature_size = model_channels
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for nr in range(self.num_res_blocks[level]):
                layers = [
                    ResBlock(
                        ch * (1 + cat_infusion * infusion_factor),
                        time_embed_dim,
                        dropout,
                        out_channels=mult * model_channels,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        # custom code for smaller models - start
                        num_head_channels = find_denominator(ch, self.num_head_channels)
                        # custom code for smaller models - end
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    if legacy:
                        # num_heads = 1
                        dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
                    if exists(disable_self_attentions):
                        disabled_sa = disable_self_attentions[level]
                    else:
                        disabled_sa = False

                    if not exists(num_attention_blocks) or nr < num_attention_blocks[level]:
                        layers.append(
                            AttentionBlock(
                                ch,
                                use_checkpoint=use_checkpoint,
                                num_heads=num_heads,
                                num_head_channels=dim_head,
                                use_new_attention_order=use_new_attention_order,
                            ) if not use_spatial_transformer else SpatialTransformer(
                                ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim,
                                disable_self_attn=disabled_sa, use_linear=use_linear_in_transformer,
                                use_checkpoint=use_checkpoint
                            )
                        )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch * (1 + (cat_infusion - seq_factor) * infusion_factor),
                            # * (infusion_factor - seq_factor),
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(
                            ch * (1 + (cat_infusion - seq_factor) * infusion_factor),
                            # * (infusion_factor - seq_factor),
                            conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        if num_head_channels == -1:
            dim_head = ch // num_heads
        else:
            num_heads = ch // num_head_channels
            dim_head = num_head_channels
        if legacy:
            # num_heads = 1
            dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch * (1 + cat_infusion * infusion_factor),
                time_embed_dim,
                dropout,
                out_channels=ch,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            AttentionBlock(
                ch,
                use_checkpoint=use_checkpoint,
                num_heads=num_heads,
                num_head_channels=dim_head,
                use_new_attention_order=use_new_attention_order,
            ) if not use_spatial_transformer else SpatialTransformer(  # always uses a self-attn
                            ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim,
                            disable_self_attn=disable_middle_self_attn, use_linear=use_linear_in_transformer,
                            use_checkpoint=use_checkpoint
                        ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self._feature_size += ch

        if guiding == 'full':
            self.output_blocks = nn.ModuleList([])
            for level, mult in list(enumerate(channel_mult))[::-1]:
                for i in range(self.num_res_blocks[level] + 1):
                    ich = input_block_chans.pop()
                    layers = [
                        ResBlock(
                            # ch + ich,
                            # ich + ch if level and i == num_res_blocks and two_stream_mode == 'sequential' else ich + ch * (infusion_factor),
                            ich + ch if level and i == num_res_blocks and two_stream_mode == 'sequential' else ich + ch * (1 + cat_infusion * infusion_factor),
                            time_embed_dim,
                            dropout,
                            out_channels=model_channels * mult,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                        )
                    ]
                    ch = model_channels * mult
                    if ds in attention_resolutions:
                        if num_head_channels == -1:
                            dim_head = ch // num_heads
                        else:
                            num_heads = ch // num_head_channels
                            dim_head = num_head_channels
                        if legacy:
                            # num_heads = 1
                            dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
                        if exists(disable_self_attentions):
                            disabled_sa = disable_self_attentions[level]
                        else:
                            disabled_sa = False

                        if not exists(num_attention_blocks) or i < num_attention_blocks[level]:
                            layers.append(
                                AttentionBlock(
                                    ch,
                                    use_checkpoint=use_checkpoint,
                                    num_heads=num_heads_upsample,
                                    num_head_channels=dim_head,
                                    use_new_attention_order=use_new_attention_order,
                                ) if not use_spatial_transformer else SpatialTransformer(
                                    ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim,
                                    disable_self_attn=disabled_sa, use_linear=use_linear_in_transformer,
                                    use_checkpoint=use_checkpoint
                                )
                            )
                    if level and i == self.num_res_blocks[level]:
                        out_ch = ch
                        layers.append(
                            ResBlock(
                                ch,
                                time_embed_dim,
                                dropout,
                                out_channels=out_ch,
                                dims=dims,
                                use_checkpoint=use_checkpoint,
                                use_scale_shift_norm=use_scale_shift_norm,
                                up=True,
                            )
                            if resblock_updown
                            else Upsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                        )
                        ds //= 2
                    self.output_blocks.append(TimestepEmbedSequential(*layers))
                    self._feature_size += ch


class ControlledUNetModelFixed(nn.Module):
    """
    The full UNet model with attention and timestep embedding.
    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param num_res_blocks: number of residual blocks per downsample.
    :param attention_resolutions: a collection of downsample rates at which
        attention will take place. May be a set, list, or tuple.
        For example, if this contains 4, then at 4x downsampling, attention
        will be used.
    :param dropout: the dropout probability.
    :param channel_mult: channel multiplier for each level of the UNet.
    :param conv_resample: if True, use learned convolutions for upsampling and
        downsampling.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param num_classes: if specified (as an int), then this model will be
        class-conditional with `num_classes` classes.
    :param use_checkpoint: use gradient checkpointing to reduce memory usage.
    :param num_heads: the number of attention heads in each attention layer.
    :param num_heads_channels: if specified, ignore num_heads and instead use
                               a fixed channel width per attention head.
    :param num_heads_upsample: works with num_heads to set a different number
                               of heads for upsampling. Deprecated.
    :param use_scale_shift_norm: use a FiLM-like conditioning mechanism.
    :param resblock_updown: use residual blocks for up/downsampling.
    :param use_new_attention_order: use a different attention pattern for potentially
                                    increased efficiency.
    """

    def __init__(
        self,
        image_size,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        num_classes=None,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=-1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_new_attention_order=False,
        use_spatial_transformer=False,    # custom transformer support
        transformer_depth=1,              # custom transformer support
        context_dim=None,                 # custom transformer support
        n_embed=None,                     # custom support for prediction of discrete ids into codebook of first stage vq model
        legacy=True,
        disable_self_attentions=None,
        num_attention_blocks=None,
        disable_middle_self_attn=False,
        use_linear_in_transformer=False,
        infusion2control='cat',         # how to infuse intermediate information into the control net? {'add', 'cat', None}
        guiding='encoder',              # use just encoder for control or the whole encoder + decoder net? {'encoder', 'encoder_double', 'full'}
        two_stream_mode='cross',        # mode for the two stream infusion. {'cross', 'sequential'}
        control_model_ratio=1.0,
        fixed=False,
    ):
        super().__init__()
        if use_spatial_transformer:
            assert context_dim is not None, 'Fool!! You forgot to include the dimension of your cross-attention conditioning...'

        if context_dim is not None:
            assert use_spatial_transformer, 'Fool!! You forgot to use the spatial transformer for your cross-attention conditioning...'
            from omegaconf.listconfig import ListConfig
            if type(context_dim) == ListConfig:
                context_dim = list(context_dim)

        self.infusion2control = infusion2control
        # infusion_factor = 2 if infusion2control == 'cat' else 1
        # infusion_factor = int(1 / control_model_ratio)
        infusion_factor = 1 / control_model_ratio
        if not fixed:
            infusion_factor = int(infusion_factor)

        cat_infusion = 1 if infusion2control == 'cat' else 0

        self.guiding = guiding
        self.two_stage_mode = two_stream_mode
        # seq_factor = 1 if two_stream_mode == 'sequential' and infusion2control == 'cat' else 0
        seq_factor = 1 if two_stream_mode == 'sequential' and infusion2control == 'cat' else 0

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        if num_heads == -1:
            assert num_head_channels != -1, 'Either num_heads or num_head_channels has to be set'

        if num_head_channels == -1:
            assert num_heads != -1, 'Either num_heads or num_head_channels has to be set'

        self.image_size = image_size
        self.in_channels = in_channels
        # self.model_channels = model_channels
        self.out_channels = out_channels
        if isinstance(num_res_blocks, int):
            self.num_res_blocks = len(channel_mult) * [num_res_blocks]
        else:
            if len(num_res_blocks) != len(channel_mult):
                raise ValueError("provide num_res_blocks either as an int (globally constant) or "
                                 "as a list/tuple (per-level) with the same length as channel_mult")
            self.num_res_blocks = num_res_blocks
        if disable_self_attentions is not None:
            # should be a list of booleans, indicating whether to disable self-attention in TransformerBlocks or not
            assert len(disable_self_attentions) == len(channel_mult)
        if num_attention_blocks is not None:
            assert len(num_attention_blocks) == len(self.num_res_blocks)
            assert all(map(lambda i: self.num_res_blocks[i] >= num_attention_blocks[i], range(len(num_attention_blocks))))
            print(f"Constructor of UNetModel received num_attention_blocks={num_attention_blocks}. "
                  f"This option has LESS priority than attention_resolutions {attention_resolutions}, "
                  f"i.e., in cases where num_attention_blocks[i] > 0 but 2**i not in attention_resolutions, "
                  f"attention will still not be set.")

        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        self.predict_codebook_ids = n_embed is not None

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        model_channels = max(1, int(model_channels * control_model_ratio))
        self.model_channels = model_channels
        self.control_model_ratio = control_model_ratio

        if self.num_classes is not None:
            if isinstance(self.num_classes, int):
                self.label_emb = nn.Embedding(num_classes, time_embed_dim)
            elif self.num_classes == "continuous":
                print("setting up linear c_adm embedding layer")
                self.label_emb = nn.Linear(1, time_embed_dim)
            else:
                raise ValueError()

        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential(
                    conv_nd(dims, in_channels, model_channels, 3, padding=1)
                )
            ]
        )
        self._feature_size = model_channels
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for nr in range(self.num_res_blocks[level]):
                layers = [
                    ResBlock(
                        int(ch * (1 + cat_infusion * infusion_factor)),
                        time_embed_dim,
                        dropout,
                        out_channels=mult * model_channels,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = mult * model_channels
                # print(f'\n\n[INPUT BLOCKS {ch}]\n\n')
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = max(num_heads, ch // num_heads)
                    else:
                        # custom code for smaller models - start
                        num_head_channels = find_denominator(ch, min(ch, self.num_head_channels))
                        # custom code for smaller models - end
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    if legacy:
                        # num_heads = 1
                        dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
                    if exists(disable_self_attentions):
                        disabled_sa = disable_self_attentions[level]
                    else:
                        disabled_sa = False

                    if not exists(num_attention_blocks) or nr < num_attention_blocks[level]:
                        # print(f'[INPUT ATTENTION HEAD DIM {dim_head}, CH {ch}, num_heads {num_heads}, max {max(num_heads, dim_head)}]')
                        layers.append(
                            AttentionBlock(
                                ch,
                                use_checkpoint=use_checkpoint,
                                num_heads=num_heads,
                                num_head_channels=dim_head,
                                use_new_attention_order=use_new_attention_order,
                            ) if not use_spatial_transformer else SpatialTransformer(
                                ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim,
                                disable_self_attn=disabled_sa, use_linear=use_linear_in_transformer,
                                use_checkpoint=use_checkpoint
                            )
                        )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            int(ch * (1 + (cat_infusion - seq_factor) * infusion_factor)),
                            # * (infusion_factor - seq_factor),
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(
                            int(ch * (1 + (cat_infusion - seq_factor) * infusion_factor)),
                            # * (infusion_factor - seq_factor),
                            conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        if num_head_channels == -1:
            dim_head = max(num_heads, ch // num_heads)
            # dim_head = ch // num_heads
        else:
            # num_heads = ch // num_head_channels
            # dim_head = num_head_channels
            # custom code for smaller models - start
            num_head_channels = find_denominator(ch, min(ch, self.num_head_channels))
            # custom code for smaller models - end
            num_heads = ch // num_head_channels
            dim_head = num_head_channels
        if legacy:
            # num_heads = 1
            dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
        # print(f'[MIDBLOCK ATTENTION HEAD DIM {dim_head}, CH {ch}, num_heads {num_heads}, max {max(num_heads, dim_head)}]')
        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                int(ch * (1 + cat_infusion * infusion_factor)),
                time_embed_dim,
                dropout,
                out_channels=ch,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            AttentionBlock(
                ch,
                use_checkpoint=use_checkpoint,
                num_heads=num_heads,
                num_head_channels=dim_head,
                use_new_attention_order=use_new_attention_order,
            ) if not use_spatial_transformer else SpatialTransformer(  # always uses a self-attn
                            ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim,
                            disable_self_attn=disable_middle_self_attn, use_linear=use_linear_in_transformer,
                            use_checkpoint=use_checkpoint
                        ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self._feature_size += ch

        if guiding == 'full':
            self.output_blocks = nn.ModuleList([])
            for level, mult in list(enumerate(channel_mult))[::-1]:
                for i in range(self.num_res_blocks[level] + 1):
                    ich = input_block_chans.pop()
                    layers = [
                        ResBlock(
                            # ch + ich,
                            # ich + ch if level and i == num_res_blocks and two_stream_mode == 'sequential' else ich + ch * (infusion_factor),
                            ich + ch if level and i == num_res_blocks and two_stream_mode == 'sequential' else int(ich + ch * (1 + cat_infusion * infusion_factor)),
                            time_embed_dim,
                            dropout,
                            out_channels=model_channels * mult,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                        )
                    ]
                    ch = model_channels * mult
                    if ds in attention_resolutions:
                        if num_head_channels == -1:
                            dim_head = ch // num_heads
                        else:
                            num_heads = ch // num_head_channels
                            dim_head = num_head_channels
                        if legacy:
                            # num_heads = 1
                            dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
                        if exists(disable_self_attentions):
                            disabled_sa = disable_self_attentions[level]
                        else:
                            disabled_sa = False

                        if not exists(num_attention_blocks) or i < num_attention_blocks[level]:
                            layers.append(
                                AttentionBlock(
                                    ch,
                                    use_checkpoint=use_checkpoint,
                                    num_heads=num_heads_upsample,
                                    num_head_channels=dim_head,
                                    # num_head_channels=max(num_heads, dim_head),
                                    use_new_attention_order=use_new_attention_order,
                                ) if not use_spatial_transformer else SpatialTransformer(
                                    ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim,
                                    # ch, num_heads, max(num_heads, dim_head), depth=transformer_depth, context_dim=context_dim,
                                    disable_self_attn=disabled_sa, use_linear=use_linear_in_transformer,
                                    use_checkpoint=use_checkpoint
                                )
                            )
                    if level and i == self.num_res_blocks[level]:
                        out_ch = ch
                        layers.append(
                            ResBlock(
                                ch,
                                time_embed_dim,
                                dropout,
                                out_channels=out_ch,
                                dims=dims,
                                use_checkpoint=use_checkpoint,
                                use_scale_shift_norm=use_scale_shift_norm,
                                up=True,
                            )
                            if resblock_updown
                            else Upsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                        )
                        ds //= 2
                    self.output_blocks.append(TimestepEmbedSequential(*layers))
                    self._feature_size += ch


class ControlledEncoder(nn.Module):
    def __init__(self, *, ch, out_ch, ch_mult=(1, 2, 4, 8), num_res_blocks,
                 attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels,
                 resolution, z_channels, double_z=True, use_linear_attn=False, attn_type="vanilla",
                 infusion2control='cat', two_stream_mode='cross', control_model_ratio=1.0,
                 **ignore_kwargs):
        super().__init__()
        if use_linear_attn:
            attn_type = "linear"

        self.infusion2control = infusion2control
        infusion_factor = int(1 / control_model_ratio)
        cat_infusion = 1 if infusion2control == 'cat' else 0
        self.two_stream_mode = two_stream_mode
        seq_factor = 1 if two_stream_mode == 'sequential' and infusion2control == 'cat' else 0
        self.control_model_ratio = control_model_ratio

        ch = int(ch * control_model_ratio)
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels

        # downsampling
        self.conv_in = torch.nn.Conv2d(in_channels,
                                       self.ch,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        curr_res = resolution
        in_ch_mult = (1,)+tuple(ch_mult)
        self.in_ch_mult = in_ch_mult
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch*in_ch_mult[i_level]
            block_out = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(
                    ae_model.ResnetBlock(
                        in_channels=block_in * (1 + cat_infusion * infusion_factor),
                        out_channels=block_out,
                        temb_channels=self.temb_ch,
                        dropout=dropout
                    )
                )
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(ae_model.make_attn(block_in, attn_type=attn_type))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions-1:
                down.downsample = ae_model.Downsample(
                    block_in * (1 + (cat_infusion - seq_factor) * infusion_factor),
                    resamp_with_conv
                    )
                curr_res = curr_res // 2
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ae_model.ResnetBlock(
            in_channels=block_in * (1 + cat_infusion * infusion_factor),
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout)
        self.mid.attn_1 = ae_model.make_attn(block_in, attn_type=attn_type)
        self.mid.block_2 = ae_model.ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout)

        # end
        self.norm_out = ae_model.Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in,
                                        2*z_channels if double_z else z_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x):
        # timestep embedding
        temb = None

        # downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions-1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        h = hs[-1]
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # end
        h = self.norm_out(h)
        h = ae_model.nonlinearity(h)
        h = self.conv_out(h)
        return h


class ControlledDecoder(nn.Module):
    def __init__(self, *, ch, out_ch, ch_mult=(1, 2, 4, 8), num_res_blocks,
                 attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels,
                 resolution, z_channels, give_pre_end=False, tanh_out=False, use_linear_attn=False,
                 attn_type="vanilla",
                 infusion2control='cat', two_stream_mode='cross', control_model_ratio=1.0,
                 control_freq='block-wise',
                 **ignorekwargs):
        super().__init__()
        if use_linear_attn:
            attn_type = "linear"

        self.control_freq = control_freq
        self.infusion2control = infusion2control
        infusion_factor = int(1 / control_model_ratio)
        cat_infusion = 1 if infusion2control == 'cat' else 0
        self.two_stream_mode = two_stream_mode
        seq_factor = 1 if two_stream_mode == 'sequential' and infusion2control == 'cat' else 0
        self.control_model_ratio = control_model_ratio

        ch = int(ch * control_model_ratio)
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.give_pre_end = give_pre_end
        self.tanh_out = tanh_out

        # compute in_ch_mult, block_in and curr_res at lowest res
        in_ch_mult = (1,)+tuple(ch_mult)
        block_in = ch*ch_mult[self.num_resolutions-1]
        curr_res = resolution // 2**(self.num_resolutions-1)
        self.z_shape = (1, z_channels, curr_res, curr_res)
        print("Working with z of shape {} = {} dimensions.".format(
            self.z_shape, np.prod(self.z_shape)))

        # z to block_in
        self.conv_in = torch.nn.Conv2d(z_channels,
                                       block_in,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ae_model.ResnetBlock(
            in_channels=block_in * (1 + cat_infusion * infusion_factor),
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout)
        self.mid.attn_1 = ae_model.make_attn(
            block_in,
            attn_type=attn_type
        )
        self.mid.block_2 = ae_model.ResnetBlock(
            in_channels=block_in if control_freq == 'block-wise' else block_in * (1 + cat_infusion * infusion_factor),
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks+1):
                # print(f'i_block {i_block}\tcontrol_freq {control_freq}\t')
                block.append(ae_model.ResnetBlock(
                    # in_channels=block_in
                    # in_channels=block_in if i_level and i_block == num_res_blocks and two_stream_mode == 'sequential' else block_in * (1 + cat_infusion * infusion_factor),
                    in_channels=block_in if i_block and control_freq == 'block-wise' else block_in * (1 + cat_infusion * infusion_factor),
                    out_channels=block_out,
                    temb_channels=self.temb_ch,
                    dropout=dropout
                    ))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(ae_model.make_attn(
                        block_in,
                        attn_type=attn_type
                    ))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = ae_model.Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up)  # prepend to get consistent order

        # end
        self.norm_out = ae_model.Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in,
                                        out_ch,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, z):
        # assert z.shape[1:] == self.z_shape[1:]
        self.last_z_shape = z.shape

        # timestep embedding
        temb = None

        # z to block_in
        h = self.conv_in(z)

        # middle
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks+1):
                h = self.up[i_level].block[i_block](h, temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        if self.give_pre_end:
            return h

        h = self.norm_out(h)
        h = ae_model.nonlinearity(h)
        h = self.conv_out(h)
        if self.tanh_out:
            h = torch.tanh(h)
        return h


class TwoStreamControlledDecoder(nn.Module):

    def __init__(
        self,
        config,
        hint_channels=3,
        control_model_ratio=1.0,
        control_freq='block-wise',  # correction frequency: block-wise or module-wise
        infusion2control='cat',
        infusion2base='add',
            ):
        super().__init__()

        self.cDec = ControlledDecoder(
            infusion2control=infusion2control, control_model_ratio=control_model_ratio,
            control_freq=control_freq, **config)
        self.base = Decoder(**config)

        self.input_hint_block = TimestepEmbedSequential(
            conv_nd(2, hint_channels, 16, 3, padding=1),
            nn.SiLU(),
            conv_nd(2, 16, 16, 3, padding=1),
            nn.SiLU(),
            conv_nd(2, 16, 32, 3, padding=1, stride=2),
            nn.SiLU(),
            conv_nd(2, 32, 32, 3, padding=1),
            nn.SiLU(),
            conv_nd(2, 32, 96, 3, padding=1, stride=2),
            nn.SiLU(),
            conv_nd(2, 96, 96, 3, padding=1),
            nn.SiLU(),
            conv_nd(2, 96, 256, 3, padding=1, stride=2),
            nn.SiLU(),
            zero_module(conv_nd(2, 256, config['z_channels'], 3, padding=1))
        )

        self.control_model_ratio = control_model_ratio
        self.control_freq = control_freq
        self.infusion2control = infusion2control
        self.infusion2base = infusion2base

        self.zero_ins = nn.ModuleDict(

        )
        self.zero_outs = nn.ModuleDict()

        ch_inout_base = {'conv_in': [], 'mid': [], 'up': []}
        ch_inout_ctr = {'conv_in': [], 'mid': [], 'up': []}

        ######## gather channel sizes ########
        # input layer
        ch_inout_base['conv_in'] = (self.base.conv_in.in_channels, self.base.conv_in.out_channels)
        ch_inout_ctr['conv_in'] = (self.cDec.conv_in.in_channels, self.cDec.conv_in.out_channels)

        up_blocks_base = list(self.base.up.children())
        up_blocks_base.reverse()
        up_blocks_ctr = list(self.cDec.up.children())
        up_blocks_ctr.reverse()

        if control_freq == 'block-wise':
            #### block-wise ####
            # mid block
            ch_inout_base['mid'] = [(list(self.base.mid.children())[0].in_channels, list(self.base.mid.children())[-1].out_channels)]
            ch_inout_ctr['mid'] = [(list(self.cDec.mid.children())[0].in_channels, list(self.cDec.mid.children())[-1].out_channels)]

            # up block
            for block in up_blocks_base:
                ch_inout_base['up'].append([list(block.block.children())[0].in_channels, list(block.block.children())[-1].out_channels])
            for block in up_blocks_ctr:
                ch_inout_ctr['up'].append([list(block.block.children())[0].in_channels, list(block.block.children())[-1].out_channels])

        elif control_freq == 'module-wise':
            #### module-wise ####
            # mid block
            for module in self.base.mid.children():
                ch_in = module.in_channels
                if hasattr(module, 'out_channels'):
                    ch_out = module.out_channels
                else:
                    continue
                # ch_out = module.out_channels if hasattr(module, 'out_channels') else ch_in
                ch_inout_base['mid'].append([ch_in, ch_out])
            for module in self.cDec.mid.children():
                ch_in = module.in_channels
                if hasattr(module, 'out_channels'):
                    ch_out = module.out_channels
                else:
                    continue
                # ch_out = module.out_channels if hasattr(module, 'out_channels') else ch_in
                ch_inout_ctr['mid'].append([ch_in, ch_out])

            # up block
            for block in up_blocks_base:
                for module in block.block.children():
                    ch_in = module.in_channels
                    ch_out = module.out_channels if hasattr(module, 'out_channels') else ch_in
                    ch_inout_base['up'].append([ch_in, ch_out])
            for block in up_blocks_ctr:
                for module in block.block.children():
                    ch_in = module.in_channels
                    ch_out = module.out_channels if hasattr(module, 'out_channels') else ch_in
                    ch_inout_ctr['up'].append([ch_in, ch_out])
        else:
            raise NotImplementedError()

        self.ch_inout_base = ch_inout_base
        self.ch_inout_ctr = ch_inout_ctr

        ######## initialise infusion connections ########
        # input layer
        if infusion2control == 'add':
            self.zero_ins['conv_in'] = self.make_zero_conv(
                ch_inout_base['conv_in'][1], ch_inout_ctr['conv_in'][1]
            )
        elif infusion2control == 'cat':
            self.zero_ins['conv_in'] = self.make_zero_conv(
                ch_inout_base['conv_in'][1], ch_inout_base['conv_in'][1]
            )

        if infusion2base == 'add':
            self.zero_outs['conv_in'] = self.make_zero_conv(
                ch_inout_ctr['conv_in'][1], ch_inout_base['conv_in'][1]
            )
        elif infusion2base == 'cat':
            raise NotImplementedError()

        # mid block
        self.zero_ins['mid'] = nn.ModuleList([])
        self.zero_outs['mid'] = nn.ModuleList([])

        for ch_base, ch_ctr in zip(ch_inout_base['mid'], ch_inout_ctr['mid']):
            if infusion2control == 'add':
                self.zero_ins['mid'].append(
                    self.make_zero_conv(ch_base[1], ch_ctr[1])
                )
            elif infusion2control == 'cat':
                self.zero_ins['mid'].append(
                    self.make_zero_conv(ch_base[1], ch_base[1])
                )

            if infusion2base == 'add':
                self.zero_outs['mid'].append(
                    self.make_zero_conv(ch_ctr[1], ch_base[1])
                )
            elif infusion2base == 'cat':
                raise NotImplementedError()

        # up block
        self.zero_ins['up'] = nn.ModuleList([])
        self.zero_outs['up'] = nn.ModuleList([])

        for ch_base, ch_ctr in zip(ch_inout_base['up'], ch_inout_ctr['up']):
            if infusion2control == 'add':
                self.zero_ins['up'].append(
                    self.make_zero_conv(ch_base[1], ch_ctr[1])
                )
            elif infusion2control == 'cat':
                self.zero_ins['up'].append(
                    self.make_zero_conv(ch_base[1], ch_base[1])
                )

            if infusion2base == 'add':
                self.zero_outs['up'].append(
                    self.make_zero_conv(ch_ctr[1], ch_base[1])
                )
            elif infusion2base == 'cat':
                raise NotImplementedError()

    def make_zero_conv(self, in_channels, out_channels=None):
        self.in_channels = in_channels
        self.out_channels = out_channels or in_channels
        return TimestepEmbedSequential(
            zero_module(nn.Conv2d(in_channels, out_channels, 1, padding=0))
            )

    def forward(self, z, hint=None):
        # assert z.shape[1:] == self.base.z_shape[1:]
        self.base.last_z_shape = z.shape

        if hint is None:
            return self.base(z)

        # timestep embedding
        temb = None
        infusion_in_mid = iter(self.zero_ins['mid'])
        infusion_out_mid = iter(self.zero_outs['mid'])
        infusion_in_up = iter(self.zero_ins['up'])
        infusion_out_up = iter(self.zero_outs['up'])

        # z to block_in
        hint_guide = self.input_hint_block(hint, temb)

        h_base = self.base.conv_in(z)
        h_ctr = self.cDec.conv_in(z + hint_guide)

        corr_in = self.zero_ins['conv_in'](h_base, temb)
        corr_out = self.zero_outs['conv_in'](h_ctr, temb)

        h_base = h_base + corr_out
        if self.infusion2control == 'add':
            h_ctr = h_ctr + corr_in
        elif self.infusion2control == 'cat':
            h_ctr = torch.cat([h_ctr, corr_in], dim=1)

        # middle
        h_base = self.base.mid.block_1(h_base, temb)
        h_ctr = self.cDec.mid.block_1(h_ctr, temb)

        h_base = self.base.mid.attn_1(h_base)
        h_ctr = self.cDec.mid.attn_1(h_ctr)

        if self.control_freq == 'module-wise':
            corr_in = next(infusion_in_mid)(h_base, temb)
            corr_out = next(infusion_out_mid)(h_ctr, temb)

            h_base = h_base + corr_out
            if self.infusion2control == 'add':
                h_ctr = h_ctr + corr_in
            elif self.infusion2control == 'cat':
                h_ctr = torch.cat([h_ctr, corr_in], dim=1)
        h_base = self.base.mid.block_2(h_base, temb)
        h_ctr = self.cDec.mid.block_2(h_ctr, temb)

        corr_in = next(infusion_in_mid)(h_base, temb)
        corr_out = next(infusion_out_mid)(h_ctr, temb)

        h_base = h_base + corr_out
        if self.infusion2control == 'add':
            h_ctr = h_ctr + corr_in
        elif self.infusion2control == 'cat':
            h_ctr = torch.cat([h_ctr, corr_in], dim=1)

        # upsampling
        for i_level in reversed(range(self.base.num_resolutions)):
            for i_block in range(self.base.num_res_blocks+1):

                h_base = self.base.up[i_level].block[i_block](h_base, temb)
                h_ctr = self.cDec.up[i_level].block[i_block](h_ctr, temb)
                if len(self.base.up[i_level].attn) > 0:
                    h_base = self.base.up[i_level].attn[i_block](h_base)

                if self.control_freq == 'module-wise' and i_block != self.base.num_res_blocks:
                    corr_in = next(infusion_in_up)(h_base, temb)
                    corr_out = next(infusion_out_up)(h_ctr, temb)

                    h_base = h_base + corr_out
                    if self.infusion2control == 'add':
                        h_ctr = h_ctr + corr_in
                    elif self.infusion2control == 'cat':
                        h_ctr = torch.cat([h_ctr, corr_in], dim=1)

            if i_level != 0:
                h_base = self.base.up[i_level].upsample(h_base)
                h_ctr = self.cDec.up[i_level].upsample(h_ctr)

            corr_in = next(infusion_in_up)(h_base, temb)
            corr_out = next(infusion_out_up)(h_ctr, temb)

            h_base = h_base + corr_out
            if self.infusion2control == 'add':
                h_ctr = h_ctr + corr_in
            elif self.infusion2control == 'cat':
                h_ctr = torch.cat([h_ctr, corr_in], dim=1)

        # end
        if self.base.give_pre_end:
            return h_base

        h_base = self.base.norm_out(h_base)
        h_base = ae_model.nonlinearity(h_base)
        h_base = self.base.conv_out(h_base)
        if self.base.tanh_out:
            h_base = torch.tanh(h_base)
        return h_base


class ControlledAutoencoderKL(pl.LightningModule):
    def __init__(self,
                 ddconfig,
                 lossconfig,
                 embed_dim,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 colorize_nlabels=None,
                 monitor=None,
                 ema_decay=None,
                 learn_logvar=False,
                 infusion2control='cat',        # how to infuse intermediate information into the control net? {'add', 'cat', None}
                 infusion2base='add',            # how to infuse intermediate information into the base net? {'add', 'cat'}
                 two_stream_mode='cross',        # mode for the two stream infusion. {'cross', 'sequential'}
                 control_model_ratio=1.0,        # ratio of the control model size compared to the base model. [0, 1]
                 control_freq='module-wise',
                 synch_path=None,
                 ):
        super().__init__()

        self.infusion2control = infusion2control
        self.infusion2base = infusion2base
        self.in_ch_factor = 1 if infusion2control == 'add' else 2
        self.two_stream_mode = two_stream_mode
        self.control_model_ratio = control_model_ratio
        self.control_freq = control_freq

        self.learn_logvar = learn_logvar
        self.image_key = image_key
        self.encoder = Encoder(**ddconfig)
        self.decoder = TwoStreamControlledDecoder(
            infusion2control=infusion2control,
            control_freq=control_freq,
            control_model_ratio=control_model_ratio, config=ddconfig
            )

        self.loss = instantiate_from_config(lossconfig)
        assert ddconfig["double_z"]
        self.quant_conv = torch.nn.Conv2d(2*ddconfig["z_channels"], 2*embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        self.embed_dim = embed_dim
        if colorize_nlabels is not None:
            assert type(colorize_nlabels) == int
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
        if monitor is not None:
            self.monitor = monitor

        self.use_ema = ema_decay is not None
        if self.use_ema:
            self.ema_decay = ema_decay
            assert 0. < ema_decay < 1.
            self.model_ema = LitEma(self, decay=ema_decay)
            print(f"Keeping EMAs of {len(list(self.model_ema.buffers()))}.")

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

        if synch_path is not None:
            self.synch_weights(synch_path)

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    def synch_weights(self, synch_path):
        ckpt = torch.load(synch_path)
        res_sync = self.load_state_dict(ckpt, strict=False)
        print(f'[{len(res_sync.missing_keys)} keys are missing from the model (hint processing and cross connections included)]')

    @contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            self.model_ema.store(self.parameters())
            self.model_ema.copy_to(self)
            if context is not None:
                print(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                self.model_ema.restore(self.parameters())
                if context is not None:
                    print(f"{context}: Restored training weights")

    def on_train_batch_end(self, *args, **kwargs):
        if self.use_ema:
            self.model_ema(self)

    def encode(self, x, control=None):
        h = self.encoder(x)
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)
        return posterior

    def decode(self, z, hint=None):
        z = self.post_quant_conv(z)
        dec = self.decoder(z, hint)
        return dec

    def forward(self, input, hint, sample_posterior=True):
        posterior = self.encode(input)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        dec = self.decode(z, hint)
        return dec, posterior

    def get_input(self, batch, k):
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format).float()

        hint = batch['hint']
        if len(hint.shape) == 3:
            hint = hint[..., None]
        hint = hint.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format).float()

        return x, hint

    def training_step(self, batch, batch_idx, optimizer_idx):
        inputs, hint = self.get_input(batch, self.image_key)
        reconstructions, posterior = self(inputs, hint)

        if optimizer_idx == 0:
            # train encoder+decoder+logvar
            aeloss, log_dict_ae = self.loss(inputs, reconstructions, posterior, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")
            self.log("aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=False)
            return aeloss

        if optimizer_idx == 1:
            # train the discriminator
            discloss, log_dict_disc = self.loss(inputs, reconstructions, posterior, optimizer_idx, self.global_step,
                                                last_layer=self.get_last_layer(), split="train")

            self.log("discloss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=False)
            return discloss

    def validation_step(self, batch, batch_idx):
        log_dict = self._validation_step(batch, batch_idx)
        with self.ema_scope():
            log_dict_ema = self._validation_step(batch, batch_idx, postfix="_ema")
        return log_dict

    def _validation_step(self, batch, batch_idx, postfix=""):
        inputs, hint = self.get_input(batch, self.image_key)
        reconstructions, posterior = self(inputs, hint)
        aeloss, log_dict_ae = self.loss(inputs, reconstructions, posterior, 0, self.global_step,
                                        last_layer=self.get_last_layer(), split="val"+postfix)

        discloss, log_dict_disc = self.loss(inputs, reconstructions, posterior, 1, self.global_step,
                                            last_layer=self.get_last_layer(), split="val"+postfix)

        self.log(f"val{postfix}/rec_loss", log_dict_ae[f"val{postfix}/rec_loss"])
        self.log_dict(log_dict_ae)
        self.log_dict(log_dict_disc)
        return self.log_dict

    def configure_optimizers(self):
        lr = self.learning_rate

        # ae_params_list = list(self.decoder.parameters())
        ae_params_list = list(self.decoder.cDec.parameters())
        ae_params_list = ae_params_list + list(self.decoder.input_hint_block.parameters())
        ae_params_list = ae_params_list + list(self.decoder.zero_ins.parameters())
        ae_params_list = ae_params_list + list(self.decoder.zero_outs.parameters())
        ae_params_list = ae_params_list + list(self.decoder.input_hint_block.parameters())

        if self.learn_logvar:
            print(f"{self.__class__.__name__}: Learning logvar")
            ae_params_list.append(self.loss.logvar)
        opt_ae = torch.optim.Adam(ae_params_list,
                                  lr=lr, betas=(0.5, 0.9))
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                    lr=lr, betas=(0.5, 0.9))
        return [opt_ae, opt_disc], []

    def get_last_layer(self):
        return self.decoder.base.conv_out.weight

    @torch.no_grad()
    def log_images(self, batch, only_inputs=False, log_ema=False, **kwargs):
        log = dict()
        x, hint = self.get_input(batch, self.image_key)
        x = x.to(self.device)
        hint = hint.to(self.device)
        if not only_inputs:
            xrec, posterior = self(x, hint)
            if x.shape[1] > 3:
                # colorize with random projection
                assert xrec.shape[1] > 3
                x = self.to_rgb(x)
                xrec = self.to_rgb(xrec)
            log["samples"] = self.decode(torch.randn_like(posterior.sample()), hint)
            log["reconstructions"] = xrec
            if log_ema or self.use_ema:
                with self.ema_scope():
                    xrec_ema, posterior_ema = self(x, hint)
                    if x.shape[1] > 3:
                        # colorize with random projection
                        assert xrec_ema.shape[1] > 3
                        xrec_ema = self.to_rgb(xrec_ema)
                    log["samples_ema"] = self.decode(torch.randn_like(posterior_ema.sample()), hint)
                    log["reconstructions_ema"] = xrec_ema
        log["inputs"] = x
        log['hints'] = 1 - hint
        return log

    def to_rgb(self, x):
        assert self.image_key == "segmentation"
        if not hasattr(self, "colorize"):
            self.register_buffer("colorize", torch.randn(3, x.shape[1], 1, 1).to(x))
        x = F.conv2d(x, weight=self.colorize)
        x = 2.*(x-x.min())/(x.max()-x.min()) - 1.
        return x


def find_denominator(number, start):
    if start >= number:
        return number
    while (start != 0):
        residual = number % start
        if residual == 0:
            return start
        start -= 1


def normalization(channels):
    """
    Make a standard normalization layer.
    :param channels: number of input channels.
    :return: an nn.Module for normalization.
    """
    if find_denominator(channels, 32) < 32:
        print(f'[USING GROUPNORM OVER LESS CHANNELS ({find_denominator(channels, 32)}) FOR {channels} CHANNELS]')
    # print(f'channels {channels} \tgroups{min(channels, 32)}')
    return GroupNorm_leq32(find_denominator(channels, 32), channels)


class GroupNorm_leq32(nn.GroupNorm):
    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)


class ResBlock(TimestepBlock):
    """
    A residual block that can optionally change the number of channels.
    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param use_checkpoint: if True, use gradient checkpointing on this module.
    :param up: if True, use this block for upsampling.
    :param down: if True, use this block for downsampling.
    """

    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channels=None,
        use_conv=False,
        use_scale_shift_norm=False,
        dims=2,
        use_checkpoint=False,
        up=False,
        down=False,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm

        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            conv_nd(dims, channels, self.out_channels, 3, padding=1),
        )

        self.updown = up or down

        if up:
            self.h_upd = Upsample(channels, False, dims)
            self.x_upd = Upsample(channels, False, dims)
        elif down:
            self.h_upd = Downsample(channels, False, dims)
            self.x_upd = Downsample(channels, False, dims)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            linear(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )
        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)
            ),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(
                dims, channels, self.out_channels, 3, padding=1
            )
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

    def forward(self, x, emb):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.
        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        return checkpoint(
            self._forward, (x, emb), self.parameters(), self.use_checkpoint
        )

    def _forward(self, x, emb):
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = th.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
        return self.skip_connection(x) + h


def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=find_denominator(in_channels, 32), num_channels=in_channels, eps=1e-6, affine=True)

# def Normalize(in_channels):
#     return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)


class SpatialTransformertest(nn.Module):
    """
    Transformer block for image-like data.
    First, project the input (aka embedding)
    and reshape to b, t, d.
    Then apply standard transformer action.
    Finally, reshape to image
    NEW: use_linear for more efficiency instead of the 1x1 convs
    """
    def __init__(self, in_channels, n_heads, d_head,
                 depth=1, dropout=0., context_dim=None,
                 disable_self_attn=False, use_linear=False,
                 use_checkpoint=True):
        super().__init__()
        if exists(context_dim) and not isinstance(context_dim, list):
            context_dim = [context_dim]
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = Normalize(in_channels)
        if not use_linear:
            self.proj_in = nn.Conv2d(in_channels,
                                     inner_dim,
                                     kernel_size=1,
                                     stride=1,
                                     padding=0)
        else:
            self.proj_in = nn.Linear(in_channels, inner_dim)

        self.transformer_blocks = nn.ModuleList(
            [BasicTransformerBlock(inner_dim, n_heads, d_head, dropout=dropout, context_dim=context_dim[d],
                                   disable_self_attn=disable_self_attn, checkpoint=use_checkpoint)
                for d in range(depth)]
        )
        if not use_linear:
            self.proj_out = zero_module(nn.Conv2d(inner_dim,
                                                  in_channels,
                                                  kernel_size=1,
                                                  stride=1,
                                                  padding=0))
        else:
            self.proj_out = zero_module(nn.Linear(in_channels, inner_dim))
        self.use_linear = use_linear

    def forward(self, x, context=None):
        # note: if no context is given, cross-attention defaults to self-attention
        if not isinstance(context, list):
            context = [context]
        b, c, h, w = x.shape
        x_in = x
        x = self.norm(x)
        if not self.use_linear:
            x = self.proj_in(x)
        x = rearrange(x, 'b c h w -> b (h w) c').contiguous()
        if self.use_linear:
            x = self.proj_in(x)
        for i, block in enumerate(self.transformer_blocks):
            x = block(x, context=context[i])
        if self.use_linear:
            x = self.proj_out(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w).contiguous()
        if not self.use_linear:
            x = self.proj_out(x)
        return x + x_in
