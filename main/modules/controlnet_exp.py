import einops
import torch
import torch as th
import torch.nn as nn
import numpy as np
import pickle
import datetime

from  video_flow_diffusion import EMA,Attention,Unet3D,ResnetBlock,Downsample
from   video_flow_diffusion import GaussianDiffusion,zero_module,conv_nd,timestep_embedding,TimestepEmbedSequential


import os

class ControlledUnetModel(Unet3D):
    def forward(self, x,timesteps=None,y=None,context_list=None,context_attn_mask_list=None, control=None, only_mid_control=False, **kwargs):
        hs = []
        with torch.no_grad():


            if self.use_extra_film_by_concat:
                emb = th.cat([self.film_emb(y)], dim=-1)
            h = x.type(self.dtype)
            context_list = []
            mask_list = []
            for module in self.input_blocks:
                h = module(h, emb,context_list,mask_list)
                hs.append(h)
            h = self.middle_block(h, emb,context_list,mask_list)

        if control is not None:
            h += control.pop()

        for i, module in enumerate(self.output_blocks):
            if only_mid_control or control is None:
                h = torch.cat([h, hs.pop()], dim=1)
            else:
                h = torch.cat([h, hs.pop() + control.pop()], dim=1)
            h = module(h, emb,context_list,mask_list)

        h = h.type(x.dtype)
        return self.out(h)


class Control_img_Net(nn.Module):
    def __init__(
            self,
            image_size,
            in_channels,
            model_channels,
            melody_channels,
            num_res_blocks,
            attention_resolutions,
            dropout=0,
            channel_mult=(1, 2, 4, 8),
            conv_resample=True,
            dims=2,
            extra_film_condition_dim=None,
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
            legacy=True,
            extra_sa_layer=True,
    ):
        super().__init__()
        self.extra_film_condition_dim = extra_film_condition_dim
        self.dims = dims
        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        self.predict_codebook_ids = n_embed is not None
        time_embed_dim = model_channels * 4
        self.use_extra_film_by_concat = self.extra_film_condition_dim is not None


        if self.extra_film_condition_dim is not None:
            self.film_emb = nn.Linear(self.extra_film_condition_dim, time_embed_dim)
           

        if context_dim is not None and not use_spatial_transformer:
            assert (
                use_spatial_transformer
            ), "Fool!! You forgot to use the spatial transformer for your cross-attention conditioning..."

        if context_dim is not None and not isinstance(context_dim, list):
            context_dim = [context_dim]
        elif context_dim is None:
            context_dim = [None]  # At least use one spatial transformer


        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        if num_heads == -1:
            assert num_head_channels != -1, 'Either num_heads or num_head_channels has to be set'

        if num_head_channels == -1:
            assert num_heads != -1, 'Either num_heads or num_head_channels has to be set'

        self.fc_layer = nn.Linear(12, 64)
 
        self.time_embed = nn.Sequential(
        )

        self.input_blocks = nn.ModuleList(
            []
        )
        self.zero_convs = nn.ModuleList([self.make_zero_conv(model_channels)])



        self._feature_size = model_channels
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1

        for level, mult in enumerate(channel_mult):

            for _ in range(num_res_blocks):
                layers = [
                    ResnetBlock(
                        ch,
                        time_embed_dim
                        if (not self.use_extra_film_by_concat)
                        else time_embed_dim * 2,
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
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    if legacy:
                        dim_head = (
                            ch // num_heads
                            if use_spatial_transformer
                            else num_head_channels
                        )

                    for context_dim_id in range(len(context_dim)):
                        layers.append(
                            Attention(
                                ch,
                                use_checkpoint=use_checkpoint,
                                num_heads=num_heads,
                                num_head_channels=dim_head,
                                use_new_attention_order=use_new_attention_order,
                            )
                        )
                self.zero_convs.append(self.make_zero_conv(ch))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(self.channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(

                        ResnetBlock(
                            ch,
                            time_embed_dim
                            if (not self.use_extra_film_by_concat)
                            else time_embed_dim * 2,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                ch = out_ch
                input_block_chans.append(ch)
                self.zero_convs.append(self.make_zero_conv(ch))
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
        middle_layers = [
            ResnetBlock(
                ch,
                time_embed_dim
                if (not self.use_extra_film_by_concat)
                else time_embed_dim * 2,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            )
        ]
        if extra_sa_layer:
            middle_layers.append()
        for context_dim_id in range(len(context_dim)):
            middle_layers.append(
                Attention(
                    ch,
                    use_checkpoint=use_checkpoint,
                    num_heads=num_heads,
                    num_head_channels=dim_head,
                    use_new_attention_order=use_new_attention_order,
                )
            )
        middle_layers.append(
            ResnetBlock(
                ch,
                time_embed_dim
                if (not self.use_extra_film_by_concat)
                else time_embed_dim * 2,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            )
        )
        self.middle_block_out = self.make_zero_conv(ch)
        self._feature_size += ch

    def make_zero_conv(self, channels):
        return TimestepEmbedSequential(zero_module(conv_nd(self.dims, channels, channels, 1, padding=0)))

    def forward(self, x, melody, timesteps, y, **kwargs):
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        emb = self.time_embed(t_emb).unsqueeze(1).contiguous()
        if self.use_extra_film_by_concat:
            emb = th.cat([emb, self.film_emb(y)], dim=-1)
        melody = self.fc_layer(melody.squeeze()).unsqueeze(1).contiguous()
        context_list = []
        mask_list = []
        guided_melody = self.input_melody_block(melody,emb,context_list,mask_list)
        outs = []
        h = x.type(self.dtype)
        for module, zero_conv in zip(self.input_blocks, self.zero_convs):
            if guided_melody is not None:
                h = module(h, emb,context_list,mask_list)
                h += guided_melody
                guided_melody = None
            else:
                h = module(h, emb,context_list,mask_list)
            outs.append(zero_conv(h, emb,context_list,mask_list))

        h = self.middle_block(h, emb,context_list,mask_list)
        outs.append(self.middle_block_out(h, emb,context_list,mask_list))

        return outs


class Control_Film_LDM():

    def __init__(self, control_stage_config, control_key, only_mid_control, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.control_key = control_key
        self.only_mid_control = only_mid_control
        self.control_scales = [1.0] * 13

    @torch.no_grad()
    def get_input(self, batch, k, bs=None, *args, **kwargs):
        x, c = super().get_input(batch, self.first_stage_key, *args, **kwargs)
        control_path = batch[self.control_key]
        control = []
        for i in control_path:
            with open(i ,'rb') as f:
                data = pickle.load(f).cpu().unsqueeze(0).contiguous()
                control.append(data)
        if bs is not None:
            control = control[:bs]
        control  = np.array(control)
        control = torch.tensor(control).contiguous().to(self.device)
        control = control.to(memory_format=torch.contiguous_format).float()
        return x, dict(c_video_concat=[c], c_concat=[control])

    def apply_model(self, x_noisy, t, cond, *args, **kwargs):
        assert isinstance(cond, dict)
        diffusion_model = self.model.diffusion_model

        cond_movie = torch.cat(cond['c_video_concat'], 1).contiguous()
        context_list, attn_mask_list = [], []
        if cond['c_concat'] is None:
            eps = diffusion_model(x=x_noisy, timesteps=t, y=cond_movie, control=None, only_mid_control=self.only_mid_control,conytext_list = context_list,attn_mask_list = attn_mask_list)
        else:
            control = self.control_model(x=x_noisy, melody=torch.cat(cond['c_concat'], 1).contiguous(), timesteps=t, y=cond_movie)
            control = [c * scale for c, scale in zip(control, self.control_scales)]
            eps = diffusion_model(x=x_noisy, timesteps=t, y=cond_movie, control=control, only_mid_control=self.only_mid_control,context_list = context_list,attn_mask_list = attn_mask_list)

        return eps


    def get_validation_folder_name(self):
        now = datetime.datetime.now()
        timestamp = now.strftime("%m-%d-%H:%M")
        return "val_%s_cfg_scale_%s_ddim_%s_n_cand_%s" % (
            timestamp,
            3.5,
            200,
            1,
        )
    EMA()




    def on_validation_epoch_start(self) -> None:
        self.validation_folder_name = self.get_validation_folder_name()
        return super().on_validation_epoch_start()

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        self.generate_sample(
            [batch],
            name=self.validation_folder_name,
            unconditional_guidance_scale=3.5,
            ddim_steps=200,
            n_gen=1,
        )


    @torch.no_grad()
    def generate_sample(
        self,
        batchs,
        ddim_steps=200,
        ddim_eta=1.0,
        x_T=None,
        n_gen=1,
        unconditional_guidance_scale=1.0,
        unconditional_conditioning=None,
        name=None,
        **kwargs,
    ):
        assert x_T is None    #None
        try:
            batchs = iter(batchs)
        except TypeError:
            raise ValueError("The first input argument should be an iterable object")

        use_ddim = ddim_steps is not None   #200
        if name is None:
            name = self.get_validation_folder_name()

        waveform_save_path = os.path.join("/data/val_outcome/", name)

        os.makedirs(waveform_save_path, exist_ok=True)
        for i, batch in enumerate(batchs):
            z, c = self.get_input(
                batch,
                self.first_stage_key,
            )   

            batch_size = z.shape[0] * n_gen   #1
            fnames = list(super().get_input(batch, "fname"))    

            samples, _ = self.sample_log(
                cond=c,
                batch_size=batch_size,
                x_T=x_T,
                ddim=use_ddim,
                ddim_steps=ddim_steps,
                eta=ddim_eta,
                unconditional_guidance_scale=unconditional_guidance_scale,
                unconditional_conditioning=unconditional_conditioning,
            )

            mel = self.decode_first_stage(samples)

            waveform = self.mel_spectrogram_to_waveform(
                mel, savepath=waveform_save_path, bs=None, name=fnames, save=False
            )
            self.save_waveform(waveform, waveform_save_path, name=fnames)
        return waveform_save_path
    
    @torch.no_grad()
    def sample_log(
        self,
        cond,
        batch_size,
        ddim,
        ddim_steps,
        unconditional_guidance_scale=1.0,
        unconditional_conditioning=None,
        mask=None,
        **kwargs,
    ):
        if mask is not None:
            shape = (self.channels, mask.size()[-2], mask.size()[-1])
        else:
            shape = (self.channels, self.latent_t_size, self.latent_f_size)

        intermediate = None
        if ddim:

            ddim_sampler = GaussianDiffusion(self)
            samples, intermediates = ddim_sampler.sample(
                ddim_steps,
                batch_size,
                shape,
                cond,
                verbose=False,
                unconditional_guidance_scale=unconditional_guidance_scale,
                unconditional_conditioning=unconditional_conditioning,
                mask=mask,
                **kwargs,
            )

        else:
            print("Use DDPM sampler")
            samples, intermediates = self.sample(
                cond=cond,
                batch_size=batch_size,
                return_intermediates=True,
                unconditional_guidance_scale=unconditional_guidance_scale,
                mask=mask,
                unconditional_conditioning=unconditional_conditioning,
                **kwargs,
            )

        return samples, intermediate

    def save_waveform(self, waveform, savepath, name="outwav"):
        for i in range(waveform.shape[0]):
            if type(name) is str:
                path = os.path.join(
                    savepath, "%s.wav" % ( name)
                )
            elif type(name) is list:
                path = os.path.join(
                    savepath,
                    "%s.wav"
                    % (
                        os.path.basename(name[i])
                        if (not ".wav" in name[i])
                        else os.path.basename(name[i]).split(".")[0]
                    ),
                )
            else:
                raise NotImplementedError
            todo_waveform = waveform[i, 0]
            todo_waveform = (
                todo_waveform / np.max(np.abs(todo_waveform))
            ) * 0.8  # Normalize the energy of the generation output

    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(self.control_model.parameters())
        if not self.sd_locked:
            params += list(self.model.diffusion_model.output_blocks.parameters())
            params += list(self.model.diffusion_model.out.parameters())
        opt = torch.optim.AdamW(params, lr=lr)
        return opt

    def low_vram_shift(self, is_diffusing):
        if is_diffusing:
            self.model = self.model.cuda()
            self.control_model = self.control_model.cuda()
            self.first_stage_model = self.first_stage_model.cpu()
            self.cond_stage_model = self.cond_stage_model.cpu()
        else:
            self.model = self.model.cpu()
            self.control_model = self.control_model.cpu()
            self.first_stage_model = self.first_stage_model.cuda()
            self.cond_stage_model = self.cond_stage_model.cuda()
