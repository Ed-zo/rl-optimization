import math
import os
import subprocess
from pathlib import Path
from typing import List

import networkx as nx
import numpy as np
import pandas as pd
import torch
from matplotlib import gridspec
from matplotlib import pyplot as plt
from torch import Tensor, tensor
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx

from env import Env


class GraphVisualizer:

    def __init__(
        self,
        output_dir: Path,
        image_duration: int = 1,
        number_of_pies: int = 3,
        top_n_probabilities: int = 10,
    ):
        self.output_dir = output_dir
        self.image_duration = image_duration
        self.number_of_pies = number_of_pies
        self.top_n_probabilities = top_n_probabilities
        self.counter = 0

        os.makedirs(output_dir, exist_ok=True)

    def save_dashboard_step(self, env: Env, softmax: Tensor, observation: Data):
        """Saves an image of the dashboard to the output directory.

        Args:
            env (Env): Environment.
            softmax (Tensor): Softmax probabilities of the actions.
            observation (Data): Current observation.
        """
        fig = self.create_dashboard(env, softmax, observation)
        fig.savefig(self.output_dir / f"image_{str(self.counter).zfill(4)}.png")
        plt.close(fig)
        self.counter += 1

    def draw_grad_pies(
        self,
        ax_pies: gridspec.SubplotSpec,
        ax_legend: plt.Axes,
        grads: List[Tensor],
        probabilities: np.array,
        n: int = 3,
    ):
        """Draws pie charts highligting which features contributed the most.

        Number of pies can be adjusted in the constructor.

        Args:
            ax_pies (gridspec.SubplotSpec): Subplot to draw the pies on.
            ax_legend (plt.Axes): Axes to draw the legend on.
            grads (List[Tensor]): List of gradients for the top n actions.
            probabilities (np.array): Array of probabilities (used only to ged node indices)
            n (int, optional): Number of pies to draw. Defaults to 3.
        """
        ratios = [1] * n

        gs_pies = gridspec.GridSpecFromSubplotSpec(
            n, 1, subplot_spec=ax_pies, height_ratios=ratios
        )

        labels = [
            "ZastavkaStart",
            "ZastavkaFinish",
            "CasStart",
            "CasFinish",
            "Vzdialenost",
            "Trvanie",
            "Pridelené vozidlo",
            "Aktuálna pozícia",
        ]

        colors = []
        nodes_indices = np.argsort(probabilities)[::-1][:n]

        for i in range(n):
            pie_sizes = grads[i] / grads[i].max()
            if torch.isnan(pie_sizes).any():
                continue
            pie_sizes, _ = torch.sort(pie_sizes, descending=True)

            ax = plt.subplot(gs_pies[i])
            wedges, _, autopct = ax.pie(
                pie_sizes, labels=None, autopct="%1d%%", pctdistance=1.15
            )
            plt.setp(autopct, fontsize=20)
            if colors == []:
                colors = [w.get_facecolor() for w in wedges]
            ax.set_title(f"{i + 1}. Node {nodes_indices[i]}", fontsize=26, loc="left")

        legend_patches = [
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=color,
                markersize=25,
                linewidth=20,
            )
            for color in colors
        ]
        ax_legend.legend(legend_patches, labels, loc="center", fontsize=26)

    def draw_graph(
        self,
        ax: plt.Axes,
        graph: Data,
        probabilities: np.array,
        current_node: int,
        n: int = 10,
    ):
        """Draws a graph with connections between the current node and top n nodes with highest probabilities.

        Args:
            ax (plt.Axes): Axes to draw the graph on.
            graph (Data): Graph to draw.
            probabilities (np.array): Array of probabilities.
            current_node (int): Current node.
            n (int, optional): Number of top probabilities to show. Defaults to 10.
        """
        positions = {}
        center = np.mean(graph.x[:, 2].numpy())
        grid_size = math.ceil(math.sqrt(graph.num_nodes))

        for i in range(graph.num_nodes):
            node = graph.x[i]
            if i == 0 or i == graph.num_nodes - 1:
                positions[i] = [node[2], center + grid_size // 2]
                continue
            col = i % grid_size

            # Position the node at the calculated row and column
            positions[i] = [node[2], col]

        # Get top n nodes with highest probabilities (excluding 0 probabilities)
        top_indicies = []
        for index in np.argsort(probabilities)[-n:][::-1]:
            if probabilities[index] > 0:
                top_indicies.append(index)

        # Generate edges to display for top n nodes
        edge_list = [(current_node, i) for i in top_indicies]

        g = to_networkx(graph)

        nx.draw(
            g,
            pos=positions,
            arrows=True,
            edgelist=edge_list,
            edge_color="black",
            cmap=plt.get_cmap("plasma"),
            node_size=500,
            node_color=probabilities,
            with_labels=True,
            ax=ax,
            width=2,
        )
        ax.set_title(f"Current node: {current_node}", fontsize=30)

    def draw_table(self, ax_table: plt.Axes, probabilities: np.array, n: int = 10):
        """Draws a table with top n probabilities.

        Args:
            ax_table (plt.Axes): Axes to draw the table on.
            probabilities (np.array): Array of probabilities.
            n (int, optional): Number of top probabilities to show. Defaults to 10.
        """
        nodes = probabilities.argsort()[::-1][:n]
        probabilities_top_n = probabilities[nodes]
        probabilities_top_n = np.round(probabilities_top_n, decimals=10)
        df = pd.DataFrame(
            {"Node": nodes, "Probability": [str(i) for i in probabilities_top_n]}
        )
        df.sort_values(by="Probability", ascending=False, inplace=True)

        table = ax_table.table(
            cellText=df.values, colLabels=df.columns, loc="center", cellLoc="center"
        )
        table.auto_set_font_size(False)
        table.set_fontsize(22)
        table.scale(1, 2)
        for key, cell in table.get_celld().items():
            if key[0] == 0:  # Hlavička tabuľky
                cell.set_fontsize(24)
                cell.set_linewidth(2)  # Nastavenie hrúbky okrajov
                cell.set_facecolor("grey")  # Nastavenie farby pozadia

    def create_dashboard(
        self, env: Env, softmax: tensor, observation: Data
    ) -> plt.Figure:
        """Creates an overview of the environment state in a simple dashboard.

        Dashboard is divided into 4 parts with the following content:
        - Graph with top n probabilities in the top left corner
        - Table with top n probabilities in the bottom left corner
        - Gradient pies in the top right corner
        - Legend for the gradient pies in the bottom right corner
        Number of pies and shown probabilities can be adjusted in the constructor.

        Args:
            env (Env): Environment.
            softmax (tensor): Softmax probabilities of the actions.
            observation (Data): Current observation.

        Returns:
            plt.Figure: Dashboard figure.
        """

        graph = env.graph
        current_node = env.last_visited_node
        probabilities = softmax.cpu().detach().numpy()[0]

        # Define the figure and subplots
        fig = plt.figure(figsize=(32, 18), tight_layout=True)
        gs = gridspec.GridSpec(2, 2, height_ratios=[26, 6], width_ratios=[14, 4])

        axis_graph = plt.subplot(gs[0, 0])
        self.draw_graph(
            axis_graph, graph, probabilities, current_node, self.top_n_probabilities
        )

        axis_pies = gs[
            0, 1
        ]  # need to send subplotspec to draw_grad_pies, we further divide the plot space
        axis_legend = plt.subplot(gs[1, 1])
        axis_legend.axis("off")
        grads = self.calculate_grads(softmax, observation, self.number_of_pies)
        self.draw_grad_pies(
            axis_pies, axis_legend, grads, probabilities, self.number_of_pies
        )

        axis_table = plt.subplot(gs[1, 0])
        axis_table.axis("off")
        self.draw_table(axis_table, probabilities, self.top_n_probabilities)

        return fig

    def calculate_grads(
        self, softmax: tensor, observation: Data, n: int = 3
    ) -> List[Tensor]:
        """Calculates gradients for the top n actions with highest probability.

        Args:
            softmax (tensor): Softmax probabilities of the actions.
            observation (Data): Current observation.
            n (int, optional): Number of gradients to calculate. Defaults to 3.

        Returns:
            List[Tensor]: List of gradients for the top n actions.
        """
        grads = []
        indicies = softmax.argsort(descending=True)[0][:n]
        for action in indicies:
            output = softmax[0][action]
            grad = torch.autograd.grad(output, observation.x, retain_graph=True)
            grad, _ = torch.max(
                grad[0].relu(), 0
            )  # select maximum value along each feature that contributes to the action
            grads.append(grad)

        return grads

    def export_video(self, dir: Path, image_duration: int = 1):
        """Exports images in the directory to a video file using ffmpeg.

        Video is exported to the same directory with the name 'output.mp4'.

        Args:
            dir (Path): Directory with .png images.
            image_duration (int, optional): Duration of each image in the video. Defaults to 1 second.

        """
        # find all .png images in the directory
        paths = list(dir.glob("*.png"))
        # sort them by name
        paths.sort(key=lambda x: int(x.stem.split("_")[-1]))
        # create txt file with structure for ffmpeg
        with open(dir / "filelist.txt", "w") as f:
            for path in paths:
                f.write(f"file '{str(path)}'\n")
                f.write(f"duration {image_duration}\n")

        cmd = [
            "ffmpeg",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(dir / "filelist.txt"),
            "-vsync",
            "vfr",
            "-pix_fmt",
            "yuv420p",
            str(dir / "output.mp4"),
        ]
        subprocess.run(cmd, check=True, text=True, capture_output=True)

        os.remove(dir / "filelist.txt")
        os.system(f"start {dir / 'output.mp4'}")
