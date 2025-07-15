import os
from collections import OrderedDict

import numpy as np
import polars as pl
import torch
import torch.nn as nn
from rich import box
from rich.console import Console
from rich.table import Table
from rich.terminal_theme import MONOKAI
from rich.text import Text
from torchvision.models.resnet import BasicBlock


class ModelSummarizer:
    """Prints a summary of the model."""

    def __init__(
        self,
        model,
        save_path=None,
        dataloader=None,
        input_size=None,
        batch_size=-1,
        device=torch.device("cuda:0"),
        dtypes=None,
    ):
        """Initializes the ModelSummarizer class.

        Args:
            model (torch.nn.Module): PyTorch model.
            dataloader (torch.utils.data.DataLoader, optional): Dataloader to infer input shape.
                If provided, `input_size` will be ignored. Defaults to None.
            input_size (tuple, optional): Input size of the model (e.g. (3, 224, 224)).
                Required if `dataloader` is not provided. Defaults to None.
            batch_size (int, optional): Batch size. Defaults to -1.
            device (torch.device, optional): Device. Defaults to torch.device('cuda:0').
            dtypes (list, optional): List of data types for each input. Defaults to None.
        """
        self.model = model
        self.dataloader = dataloader
        self.input_size = input_size
        self.batch_size = batch_size
        self.device = device
        self.dtypes = dtypes
        self.save_path = save_path

        if self.dataloader is not None:
            # Get input size from dataloader
            self.input_size = next(iter(self.dataloader))[0].shape[1:]
        elif self.input_size is None:
            raise ValueError("Please provide either 'dataloader' or 'input_size'")

    def execute(self):
        self.model.to(self.device)
        """Prints the model summary."""
        summary_dict, hooks = self._create_summary_dict()
        total_params, trainable_params = self._get_params_info(summary_dict)
        total_macs = self._get_macs_info(summary_dict)
        total_flops = self._get_flops_info(summary_dict)
        table = self._create_table(summary_dict)
        summary_str = self._create_summary_string(
            summary_dict, total_params, trainable_params, total_macs, total_flops, table
        )
        self._print_table(table, summary_str)
        df = self._table_to_df(table).write_csv(self.save_path.replace(".svg", ".csv"))
        self.model.to("cpu")
        return table, (total_params, trainable_params, total_macs, total_flops)

    def _create_summary_dict(self):
        """Creates a summary dictionary of the model.

        Returns:
            tuple: Summary dictionary and list of hooks.
        """
        if self.dtypes is None:
            self.dtypes = [torch.FloatTensor] * len(self.input_size)

        # multiple inputs to the network
        if isinstance(self.input_size, tuple):
            self.input_size = [self.input_size]

        # batch_size of 2 for batchnorm
        x = [
            torch.rand(2, *in_size).type(dtype).to(device=self.device)
            for in_size, dtype in zip(self.input_size, self.dtypes)
        ]

        # create properties
        summary_dict = OrderedDict()
        hooks = []

        # register hook
        self.model.apply(self._register_hook(summary_dict, self.batch_size, hooks))

        # make a forward pass
        self.model(*x)

        # remove these hooks
        for h in hooks:
            h.remove()

        return summary_dict, hooks

    def _get_params_info(self, summary_dict):
        """Calculates total and trainable parameters.

        Args:
            summary_dict (OrderedDict): Dictionary containing layer information.

        Returns:
            tuple: Total parameters and trainable parameters.
        """
        total_params = 0
        trainable_params = 0
        for layer in summary_dict:
            total_params += summary_dict[layer]["nb_params"]
            if "trainable" in summary_dict[layer]:
                if summary_dict[layer]["trainable"]:
                    trainable_params += summary_dict[layer]["nb_params"]
        return total_params, trainable_params

    def _get_macs_info(self, summary_dict):
        """Calculates total MACs (Multiply-Accumulate Operations).

        Args:
            summary_dict (OrderedDict): Dictionary containing layer information.

        Returns:
            int: Total number of MACs.
        """
        total_macs = 0
        for layer in summary_dict:
            total_macs += summary_dict[layer].get("macs", 0)
        return total_macs

    def _get_flops_info(self, summary_dict):
        """Calculates total FLOPs (Floating Point Operations).

        Args:
            summary_dict (OrderedDict): Dictionary containing layer information.

        Returns:
            int: Total number of FLOPs.
        """
        total_flops = 0
        for layer in summary_dict:
            total_flops += summary_dict[layer].get("flops", 0)
        return total_flops

    def _create_table(self, summary_dict):
        """Creates a Rich table with the model summary.

        Args:
            summary_dict (OrderedDict): Dictionary containing layer information.

        Returns:
            Table: Rich table with the model summary.
        """
        table = Table(title="Model Summary", box=box.ROUNDED)
        table.add_column(
            "Layer (type)", justify="right", style="light_cyan3", no_wrap=True
        )
        table.add_column("Output Shape", style="light_coral")
        table.add_column("Param #", justify="right", style="light_green")
        table.add_column("MACs", justify="right", style="light_slate_gray")
        table.add_column("FLOPs", justify="right", style="light_slate_gray")

        for layer in summary_dict:
            table.add_row(
                layer,
                str(summary_dict[layer]["output_shape"]),
                "{0:,}".format(summary_dict[layer]["nb_params"]),
                "{0:,}".format(summary_dict[layer].get("macs", 0)),
                "{0:,}".format(summary_dict[layer].get("flops", 0)),
            )
        return table

    def _create_summary_string(
        self,
        summary_dict,
        total_params,
        trainable_params,
        total_macs,
        total_flops,
        table,
    ):
        """Creates a summary string with model parameters information.

        Args:
            summary_dict (OrderedDict): Dictionary containing layer information.
            total_params (int): Total number of parameters.
            trainable_params (int): Number of trainable parameters.
            total_macs (int): Total number of MACs.
            total_flops (int): Total number of FLOPs.
            table (Table): Rich table with the model summary.

        Returns:
            str: Summary string with parameters information.
        """

        def calculate_total_output(summary_dict):
            total = 0
            for layer in summary_dict:
                output_shape = summary_dict[layer]["output_shape"]
                if isinstance(output_shape, list):  # Handle list of shapes
                    total += sum(
                        np.prod(shape)
                        for shape in output_shape
                        if isinstance(shape, (list, tuple))
                    )
                elif isinstance(output_shape, (list, tuple)):  # Single shape
                    total += np.prod(output_shape)
            return total

        # Calculate total output size
        total_output = calculate_total_output(summary_dict)

        # Assume 4 bytes/number (float32 on CUDA)
        total_input_size = abs(
            np.prod(sum(self.input_size, ())) * self.batch_size * 4.0 / (1024**2.0)
        )
        total_output_size = abs(
            2.0 * total_output * 4.0 / (1024**2.0)
        )  # x2 for gradients
        total_params_size = abs(total_params * 4.0 / (1024**2.0))
        total_size = total_params_size + total_output_size + total_input_size

        # Create dynamic separator line
        separator_line = "-" * (len(str(table).splitlines()[0]) - 3)

        # Build summary string
        summary_str = (
            f"Total params: {total_params:,}\n"
            f"Trainable params: {trainable_params:,}\n"
            f"Non-trainable params: {total_params - trainable_params:,}\n"
            f"Total MACs: {total_macs:,}\n"
            f"Total FLOPs: {total_flops:,}\n"
            f"{separator_line}\n"
            f"Input size (MB): {total_input_size:.2f}\n"
            f"Forward/backward pass size (MB): {total_output_size:.2f}\n"
            f"Params size (MB): {total_params_size:.2f}\n"
            f"Estimated Total Size (MB): {total_size:.2f}\n"
            f"{separator_line}"
        )

        return summary_str

    def _print_table(self, table, summary_str):
        """Prints the Rich table and saves it as an SVG file.

        Args:
            table (Table): Rich table with the model summary.
            summary_str (str): Summary string with parameters information.
        """
        # Add the summary string as the footer to the table
        table.caption = summary_str

        # Print the table using Rich
        console = Console(record=True)
        console.print(table)
        if self.save_path is not None:
            if not os.path.exists(os.path.dirname(self.save_path)):
                os.makedirs(os.path.dirname(self.save_path))
            console.save_svg(os.path.join(self.save_path), theme=MONOKAI)

    def _table_to_df(
        self, rich_table: Table, remove_markup: bool = True
    ) -> pl.DataFrame:
        """Converts a Rich table to a Polars DataFrame.

        Args:
            rich_table (Table): Rich table to be converted.
            remove_markup (bool, optional): Whether to remove markup tags. Defaults to True.

        Returns:
            pl.DataFrame: Converted Polars DataFrame.
        """
        return pl.DataFrame(
            {
                self._strip_tags(x.header, remove_markup): [
                    self._strip_tags(y, remove_markup) for y in x.cells
                ]
                for x in rich_table.columns
            }
        )

    def _strip_tags(self, value: str, do: bool) -> str:
        """Strips markup tags from a string.

        Args:
            value (str): String to strip tags from.
            do (bool): Whether to strip tags.

        Returns:
            str: String with tags stripped.
        """
        if do:
            return self._strip_markup_tags(value)
        else:
            return value

    def _strip_markup_tags(self, value: str) -> str:
        """Strips markup tags from a string.

        Args:
            value (str): String to strip tags from.

        Returns:
            str: String with tags stripped.
        """
        return Text.from_markup(value).plain

    def _register_hook(self, summary_dict, batch_size, hooks):
        """Registers a forward hook to the module to collect layer information.

        Args:
            summary_dict (OrderedDict): Dictionary to store layer information.
            batch_size (int): Batch size.
            hooks (list): List of registered hooks.

        Returns:
            function: Hook function to be registered.
        """

        def hook(module, input, output):
            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            module_idx = len(summary_dict)

            m_key = "%s-%i" % (class_name, module_idx + 1)
            summary_dict[m_key] = OrderedDict()

            # Handle input shape properly
            if isinstance(input, tuple) and isinstance(input[0], torch.Tensor):
                summary_dict[m_key]["input_shape"] = list(input[0].size())
            elif isinstance(input, tuple) and isinstance(input[0], (tuple, list)):
                summary_dict[m_key]["input_shape"] = [
                    list(inp.size()) if isinstance(inp, torch.Tensor) else "Non-tensor"
                    for inp in input[0]
                ]
            else:
                summary_dict[m_key]["input_shape"] = "Unknown input shape"

            summary_dict[m_key]["input_shape"][0] = batch_size

            # Handle output shape properly
            if isinstance(output, (tuple, list)):
                summary_dict[m_key]["output_shape"] = []
                for o in output:
                    if isinstance(o, torch.Tensor):
                        shape = [-1] + list(o.size())[1:]
                        summary_dict[m_key]["output_shape"].append(shape)
                    elif isinstance(o, (tuple, list)):
                        sub_shapes = [
                            (
                                list(sub_o.size())
                                if isinstance(sub_o, torch.Tensor)
                                else "Non-tensor output"
                            )
                            for sub_o in o
                        ]
                        summary_dict[m_key]["output_shape"].append(sub_shapes)
                    else:
                        summary_dict[m_key]["output_shape"].append("Non-tensor output")
            else:
                summary_dict[m_key]["output_shape"] = list(output.size())
                summary_dict[m_key]["output_shape"][0] = batch_size

            # Count parameters
            params = 0
            if hasattr(module, "weight") and hasattr(module.weight, "size"):
                params += torch.prod(torch.LongTensor(list(module.weight.size())))
                summary_dict[m_key]["trainable"] = module.weight.requires_grad
            if hasattr(module, "bias") and hasattr(module.bias, "size"):
                params += torch.prod(torch.LongTensor(list(module.bias.size())))

            summary_dict[m_key]["nb_params"] = params
            summary_dict[m_key]["macs"] = self._calculate_macs(module, output)
            summary_dict[m_key]["flops"] = self._calculate_flops(module, output)

        def register_forward_hook(module):
            if not isinstance(module, nn.Sequential) and not isinstance(
                module, nn.ModuleList
            ):
                hooks.append(module.register_forward_hook(hook))

        return register_forward_hook

    def _calculate_macs(self, module, output):
        """Calculates MACs for a given module.

        Args:
            module (nn.Module): The module to calculate MACs for.
            output (torch.Tensor): The output tensor of the module.

        Returns:
            int: Number of MACs for the module.
        """
        macs = 0
        if isinstance(module, nn.Conv2d):
            # MACs for convolution = kernel_width * kernel_height * in_channels * out_channels * output_width * output_height
            output_size = output.size()
            macs = (
                module.kernel_size[0]
                * module.kernel_size[1]
                * module.in_channels
                * module.out_channels
                * output_size[2]
                * output_size[3]
            )
        elif isinstance(module, nn.Linear):
            # MACs for linear layer = in_features * out_features
            macs = module.in_features * module.out_features
        elif isinstance(module, nn.BatchNorm2d):
            # MACs for batch norm (approximation) = 2 * num_features * output_width * output_height
            output_size = output.size()
            macs = 2 * module.num_features * output_size[2] * output_size[3]
        elif isinstance(
            module, (nn.ReLU, nn.MaxPool2d, nn.AvgPool2d, nn.AdaptiveAvgPool2d)
        ):
            # MACs for ReLU and pooling layers = output elements (since these are element-wise operations)
            macs = output.numel()
        elif isinstance(module, BasicBlock):
            # MACs for BasicBlock = sum of MACs for all sub-modules
            for sub_module in module.children():
                macs += self._calculate_macs(sub_module, output)
        return macs

    def _calculate_flops(self, module, output):
        """Calculates FLOPs for a given module.

        Args:
            module (nn.Module): The module to calculate FLOPs for.
            output (torch.Tensor): The output tensor of the module.

        Returns:
            int: Number of FLOPs for the module.
        """
        flops = 0
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            # FLOPs for convolution and linear = 2 * MACs
            flops = 2 * self._calculate_macs(module, output)
        elif isinstance(
            module,
            (nn.BatchNorm2d, nn.ReLU, nn.MaxPool2d, nn.AvgPool2d, nn.AdaptiveAvgPool2d),
        ):
            # FLOPs for batch norm, ReLU, and pooling = MACs (since these are element-wise operations)
            flops = self._calculate_macs(module, output)
        elif isinstance(module, BasicBlock):
            # FLOPs for BasicBlock = sum of FLOPs for all sub-modules
            for sub_module in module.children():
                flops += self._calculate_flops(sub_module, output)
        return flops
