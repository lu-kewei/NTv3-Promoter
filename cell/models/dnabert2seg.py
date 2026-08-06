
import torch 
from torch import nn 
from modelscope import AutoModel
from typing import List, Optional, Tuple, Union

class UpSample1D(nn.Module):
    """
    1D-UNET upsampling block.
    """

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        num_layers: int = 2,
    ):
        """
        Args:
            output_channels: number of output channels.
            activation_fn: name of the activation function to use.
                Should be one of "gelu",
                "gelu-no-approx", "relu", "swish", "silu", "sin".
            interpolation_method: Method to be used for upsampling interpolation.
                Should be one of "nearest", "linear", "cubic", "lanczos3", "lanczos5".
            num_layers: number of convolution layers.
            name: module name.
        """
        super().__init__()

        self._first_layer = [nn.ConvTranspose1d(
                in_channels=input_channels,
                out_channels=output_channels,
                kernel_size=3,
                stride=1,
                padding=1,
            )]


        self._next_layers = [
            nn.ConvTranspose1d(
                in_channels=output_channels,
                out_channels=output_channels,
                kernel_size=3,
                stride=1,
                padding=1,
            )
            for _ in range(num_layers-1)
        ]

        self.conv_layers = nn.ModuleList(self._first_layer + self._next_layers)

        self._activation_fn = nn.SiLU()



    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        for i, conv_layer in enumerate(self.conv_layers):
            x = self._activation_fn(conv_layer(x))      

        # Different order than in Haiku because the channels are changed when going 
        # from Haiku to Torch.
        x = nn.functional.interpolate(x, size=2 * x.shape[2], mode="nearest")


        return x
    
class DownSample1D(nn.Module):
    """
    1D-UNET downsampling block.
    """

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        num_layers: int = 2,
    ):
        """
        Args:
            output_channels: number of output channels.
            activation_fn: name of the activation function to use.
                Should be one of "gelu",
                "gelu-no-approx", "relu", "swish", "silu", "sin".
            num_layers: number of convolution layers.
            name: module name.
        """
        
        super().__init__()
        self.first_layer = [nn.Conv1d(
                in_channels=input_channels,
                out_channels=output_channels,
                kernel_size=3,
                stride=1,
                dilation=1,
                padding="same",
            )]
        

        self.next_layers = [
            nn.Conv1d(
                in_channels=output_channels,
                out_channels=output_channels,
                kernel_size=3,
                stride=1,
                dilation=1,
                padding="same",
            )
            for _ in range(num_layers-1)
        ]
        self.conv_layers = nn.ModuleList(self.first_layer + self.next_layers)

        self.avg_pool = nn.AvgPool1d(
            kernel_size=2,
            stride=2,
            padding=0,
        )
        self.activation_fn = nn.SiLU()


    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        for i, conv_layer in enumerate(self.conv_layers):
            x = self.activation_fn(conv_layer(x))

        hidden = x
        x = self.avg_pool(hidden)
        return x, hidden


class FinalConv1D(nn.Module):
    """
    Final output block of the 1D-UNET.
    """

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        num_layers: int = 2,
    ):
        """
        Args:
            output_channels: number of output channels.
            activation_fn: name of the activation function to use.
                Should be one of "gelu",
                "gelu-no-approx", "relu", "swish", "silu", "sin".
            num_layers: number of convolution layers.
            name: module name.
        """
        super().__init__()

        self._first_layer = [nn.Conv1d(
                in_channels=input_channels,
                out_channels=output_channels,
                kernel_size=3,
                stride=1,
                dilation=1,
                padding="same",
            )]

        self._next_layers = [
            nn.Conv1d(
                in_channels=output_channels,
                out_channels=output_channels,
                kernel_size=3,
                stride=1,
                dilation=1,
                padding="same",
            )
            for _ in range(num_layers-1)
        ]
        self.conv_layers = nn.ModuleList(self._first_layer + self._next_layers)

        self._activation_fn = nn.SiLU()



    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, conv_layer in enumerate(self.conv_layers):
            x = conv_layer(x)
            if i < len(self.conv_layers) - 1:
                x = self._activation_fn(x)
        return x
    
class UNET1DSegmentationHead(nn.Module):
    """
    1D-UNET based head to be plugged on top of a pretrained model to perform
    semantic segmentation.
    """

    def __init__(
        self,
        embed_dim: int,
        num_classes: int,
        output_channels_list: Tuple[int, ...] = (64, 128, 256),
        num_conv_layers_per_block: int = 2,
    ):
        """
        Args:
            num_classes: number of classes to segment
            output_channels_list: list of the number of output channel at each level of
                the UNET
            num_conv_layers_per_block: number of convolution layers per block.
        """
        super().__init__()
        self._num_pooling_layers = len(output_channels_list)


        downsample_input_channels_list = (embed_dim, ) + output_channels_list[:-1]

        output_channels_list_reversed = tuple(reversed(output_channels_list))
        upsample_input_channels_list = (output_channels_list[-1],) + output_channels_list_reversed
        upsample_output_channels_list =  output_channels_list_reversed

        self._downsample_blocks = nn.ModuleList([
            DownSample1D(
                input_channels= input_channels,
                output_channels=output_channels,
                num_layers=num_conv_layers_per_block,
            )
            for input_channels, output_channels in zip(downsample_input_channels_list, output_channels_list)
        ])

        self._upsample_blocks = nn.ModuleList([
            UpSample1D(
                input_channels = input_channels,
                output_channels=output_channels,
                num_layers=num_conv_layers_per_block,
            )
            for input_channels, output_channels in zip(upsample_input_channels_list, upsample_output_channels_list)
        ])

        self.final_block = FinalConv1D(
            input_channels=output_channels_list[0],
            output_channels=num_classes * 2,
            num_layers=num_conv_layers_per_block,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        if x.shape[2] % 2**self._num_pooling_layers:
            raise ValueError(
                "Input length must be divisible by the 2 to the power of"
                " number of poolign layers."
            )

        hiddens = []
        for downsample_block in self._downsample_blocks:
            x, hidden = downsample_block(x)
            hiddens.append(hidden)
        
        

        for i, (upsample_block, hidden) in enumerate(zip(self._upsample_blocks, reversed(hiddens))):
            x = upsample_block(x) + hidden
        x = self.final_block(x)
        return x


class SegmentNT(nn.Module):
    def __init__(self, dnabert2_name,embed_dim,num_layers,num_features):
        super().__init__()
        self.dnabert2 = AutoModel.from_pretrained(dnabert2_name, trust_remote_code=True)
        self.num_features = num_features
        # self.unet = UNET1DSegmentationHead(
        #     embed_dim=embed_dim,
        #     num_classes=embed_dim // 2,
        #     output_channels_list=tuple(
        #         embed_dim * (2**i) for i in range(num_layers)
        #     ),
        # )
        self.fc = nn.Linear(in_features=embed_dim, out_features=2 * self.num_features)
        # self.activation_fn = nn.SiLU()
        
    def forward(self,tokens,attention_mask,peft_flag=False):
        if peft_flag:
            embeddings = self.dnabert2(tokens, attention_mask=attention_mask)
        else:
            with torch.no_grad():
                embeddings = self.dnabert2(tokens,attention_mask=attention_mask)
        embeddings = embeddings[0][:,1:]
        # Invert the channels and sequence length channel
        # sequence_output = torch.transpose(embeddings, 2,1)

        # x = self.activation_fn(self.unet(sequence_output))

        # # Invert the channels and sequence length channel
        # x = torch.transpose(x, 2,1)

        logits = self.fc(embeddings)

        # Final reshape to have logits per nucleotides, per feature
        logits = torch.reshape(logits, (embeddings.shape[0], embeddings.shape[1], self.num_features, 2))

        outputs = {}
        outputs["logits"] = logits

        return outputs
