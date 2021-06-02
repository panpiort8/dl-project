from transformers import GPT2Model

from .ViT import *


def get_nlp_transformer(model_name: str, pretrained: bool):
    pretrained_transformer = GPT2Model.from_pretrained(model_name)
    if pretrained:
        transformer = pretrained_transformer
    else:
        transformer = GPT2Model(pretrained_transformer.config)

    return transformer, transformer.config.n_embd


class ViTEmbeddings(nn.Module):
    def __init__(self, config, max_sequence_len: int):
        super().__init__()
        self.position_embeddings = nn.Parameter(torch.zeros(1, max_sequence_len + 1, config.hidden_size))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))

    def forward(self, embeddings):
        bs = embeddings.shape[0]
        seq_len = embeddings.shape[1]
        cls_token = self.cls_token.expand(bs, -1, -1)
        embeddings = torch.cat((cls_token, embeddings), dim=1)
        embeddings = embeddings + self.position_embeddings[:, :seq_len + 1, :]
        return embeddings


class Lambda(nn.Module):
    def forward(self, output):
        x = output[0]
        return x[:, 0]


def get_vit_transformer(model_name: str, pretrained: bool, max_sequence_len: int = 1024):
    config = CONFIGS[model_name]
    model = VisionTransformer(config, num_classes=1000, img_size=384)
    if pretrained:
        model.load_from(np.load(f'{model_name}.npz'))
    transformer = nn.Sequential(ViTEmbeddings(config, max_sequence_len), model.transformer.encoder, Lambda())
    return transformer, config.hidden_size


def freeze_nlp_trans(transformer, *, freeze_ln, freeze_pos, freeze_ff, freeze_attn):
    for name, p in transformer.named_parameters():
        name = name.lower()
        if 'ln' in name:
            p.requires_grad = not freeze_ln
        elif 'wpe' in name:
            p.requires_grad = not freeze_pos
        elif 'mlp' in name:
            p.requires_grad = not freeze_ff
        elif 'attn' in name:
            p.requires_grad = not freeze_attn
        else:
            p.requires_grad = False


def freeze_vit_trans(transformer, *, freeze_ln, freeze_ff, freeze_attn):
    for name, p in transformer.named_parameters():
        name = name.lower()
        if '_norm' in name:
            p.requires_grad = not freeze_ln
        elif 'ffn.' in name:
            p.requires_grad = not freeze_ff
        elif 'attn.' in name:
            p.requires_grad = not freeze_attn
        else:
            p.requires_grad = False


class FPT(nn.Module):

    def __init__(
            self,
            input_dim,
            output_dim,
            model_name='gpt2',
            pretrained=False,
            return_last_only=True,
            use_embeddings_for_in=False,
            in_layer_sizes=None,
            out_layer_sizes=None,
            freeze_trans=True,
            freeze_in=False,
            freeze_pos=False,
            freeze_ln=False,
            freeze_attn=True,
            freeze_ff=True,
            freeze_out=False,
            dropout=0.1,
            orth_gain=1.41,
            max_sequence_len=1024,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.model_name = model_name
        self.return_last_only = return_last_only
        self.use_embeddings_for_in = use_embeddings_for_in

        self.in_layer_sizes = [] if in_layer_sizes is None else in_layer_sizes
        self.out_layer_sizes = [] if out_layer_sizes is None else out_layer_sizes
        self.dropout = dropout

        if model_name.lower().startswith('gpt2'):
            self.transformer, embedding_size = get_nlp_transformer(model_name, pretrained)
        elif model_name.lower().startswith('vit'):
            self.transformer, embedding_size = get_vit_transformer(model_name, pretrained, max_sequence_len)
        else:
            raise NotImplementedError()

        if use_embeddings_for_in:
            self.in_net = nn.Embedding(input_dim, embedding_size)
        else:
            in_layers = []
            last_output_size = input_dim
            for size in self.in_layer_sizes:
                layer = nn.Linear(last_output_size, size)
                if orth_gain is not None:
                    torch.nn.init.orthogonal_(layer.weight, gain=orth_gain)
                layer.bias.data.zero_()

                in_layers.append(layer)
                in_layers.append(nn.ReLU())
                in_layers.append(nn.Dropout(dropout))
                last_output_size = size

            final_linear = nn.Linear(last_output_size, embedding_size)
            if orth_gain is not None:
                torch.nn.init.orthogonal_(final_linear.weight, gain=orth_gain)
            final_linear.bias.data.zero_()

            in_layers.append(final_linear)
            in_layers.append(nn.Dropout(dropout))

            self.in_net = nn.Sequential(*in_layers)

        out_layers = []
        last_output_size = embedding_size
        for size in self.out_layer_sizes:
            out_layers.append(nn.Linear(last_output_size, size))
            out_layers.append(nn.ReLU())
            out_layers.append(nn.Dropout(dropout))
            last_output_size = size
        out_layers.append(nn.Linear(last_output_size, output_dim))
        self.out_net = nn.Sequential(*out_layers)

        if freeze_trans:
            if model_name.lower().startswith('gpt2'):
                freeze_nlp_trans(self.transformer,
                                 freeze_ln=freeze_ln,
                                 freeze_pos=freeze_pos,
                                 freeze_ff=freeze_ff,
                                 freeze_attn=freeze_attn)
            else:  # ViT
                freeze_vit_trans(self.transformer,
                                 freeze_ln=freeze_ln,
                                 freeze_ff=freeze_ff,
                                 freeze_attn=freeze_attn)
        if freeze_in:
            for p in self.in_net.parameters():
                p.requires_grad = False
        if freeze_out:
            for p in self.out_net.parameters():
                p.requires_grad = False

    def forward(self, x, output_attentions=False):

        orig_dim = x.shape[-1]
        if orig_dim != self.input_dim and not self.use_embeddings_for_in:
            if orig_dim % self.input_dim != 0:
                raise ValueError('dimension of x must be divisible by patch size')
            ratio = orig_dim // self.input_dim
            x = x.reshape(x.shape[0], x.shape[1] * ratio, self.input_dim)
        else:
            ratio = 1

        x = self.in_net(x)
        if self.model_name.lower().startswith('gpt2'):
            transformer_outputs = self.transformer(
                inputs_embeds=x,
                return_dict=True,
                output_attentions=output_attentions,
            )
            x = transformer_outputs.last_hidden_state

            if self.return_last_only:
                x = x[:, -ratio:]

            x = self.out_net(x)
            if self.return_last_only and ratio > 1:
                x = x.reshape(x.shape[0], x.shape[1] // ratio, ratio * self.output_dim)
            if output_attentions:
                return x, transformer_outputs.attentions
            else:
                return x
        else:  # ViT
            x = self.transformer(x)
            x = self.out_net(x)
            x = x.unsqueeze(1)
            return x
