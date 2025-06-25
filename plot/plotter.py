"""Visualization module for champion movement patterns, clustering, and heatmaps."""
from typing import Dict, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import requests
from matplotlib.animation import FuncAnimation
from pathlib import Path
from config import load_config
from utils import setup_logger

class Plotter:
    """Handles visualization of champion movement patterns, clustering, and heatmaps.

    This class manages downloading the minimap image, loading and scaling data,
    performing K-Means clustering, and generating plots (paths, clusters, heatmaps),
    scaling data to the actual size of the downloaded minimap.
    """

    COMMUNITY_DRAGON_MINIMAP_URL = (
        "https://raw.communitydragon.org/latest/game/assets/maps/info/map11/"
        "2dlevelminimap_base_baron1.png"
    )

    def __init__(self, timeline_path: str = "data/timeline.csv") -> None:
        """Initialize the Plotter with paths and dependencies.

        Args:
            timeline_path: Path to the timeline CSV file relative to project root.
                           Defaults to 'data/timeline.csv'.
        """
        self.logger = setup_logger("plotter")
        self.project_root = Path(__file__).parent # Adjust to root directory
        self.data_path = self.project_root / "data"
        self.minimap_filename = "minimap.png"
        self.minimap_path = self.data_path / self.minimap_filename
        self.timeline_path = self.project_root / timeline_path
        self.df: pd.DataFrame = None
        self.monitor_config: Dict[str, int] = None
        self.background_image: np.ndarray = None
        self.champions = None

    def check_minimap_exists(self) -> bool:
        """Check if the minimap image exists locally.

        Returns:
            True if the file exists, False otherwise.
        """
        exists = self.minimap_path.exists()
        if exists:
            self.logger.debug(f"Minimap found at {self.minimap_path}")
        else:
            self.logger.info(f"Minimap not found at {self.minimap_path}")
        return exists

    def download_minimap(self) -> None:
        """Fetch the minimap image from Community Dragon and save it locally."""
        if self.check_minimap_exists():
            self.logger.info("Minimap image already exists. Skipping download.")
            return

        self.logger.info(f"Downloading minimap from {self.COMMUNITY_DRAGON_MINIMAP_URL}")
        try:
            response = requests.get(self.COMMUNITY_DRAGON_MINIMAP_URL, timeout=10)
            response.raise_for_status()
            self.data_path.mkdir(parents=True, exist_ok=True)  # Ensure data directory exists
            with open(self.minimap_path, "wb") as file:
                file.write(response.content)
            self.logger.info(f"Minimap downloaded to {self.minimap_path}")
        except requests.RequestException as e:
            self.logger.error(f"Failed to download minimap: {e}")

    def load_and_scale_data(self) -> None:
        """Load and scale timeline data based on monitor config and minimap size."""
        config = load_config()
        self.monitor_config = {
            "top": config.get("top", 813),
            "left": config.get("left", 1655),
            "width": config.get("width", 252),
            "height": config.get("height", 252),
        }
        self.logger.debug(f"Loaded monitor config: {self.monitor_config}")

        try:
            self.df = pd.read_csv(self.timeline_path)
            self.logger.info(f"Loaded timeline data from {self.timeline_path}")
        except FileNotFoundError as e:
            self.logger.error(f"Timeline file not found at {self.timeline_path}: {e}")
            raise

        self.background_image = plt.imread(self.minimap_path)
        map_height, map_width = self.background_image.shape[:2]
        self.logger.debug(f"Minimap dimensions: {map_width}x{map_height}")

        captured_width = self.monitor_config["width"]
        captured_height = self.monitor_config["height"]
        self.df["x_scaled"] = (self.df["X"] / captured_width) * map_width
        self.df["y_scaled"] = ((captured_height - self.df["Y"]) / captured_height) * map_height
        self.df["timestamp_dt"] = pd.to_datetime(self.df["timestamp"])  # Added
        time_min = self.df["timestamp_dt"].min()
        time_range = (self.df["timestamp_dt"].max() - time_min).total_seconds()
        self.df["time_numeric"] = (self.df["timestamp_dt"] - time_min).dt.total_seconds() / time_range  # Added
        self.champions = self.df["champion"].unique()
        self.logger.info(f"Scaled data for champions: {list(self.champions)}")

    def kmeans(self, x: np.ndarray, k: int, max_iters: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """Perform K-Means clustering on the given data.

        Args:
            x: Input data array of shape (n_samples, n_features).
            k: Number of clusters.
            max_iters: Maximum iterations for convergence.

        Returns:
            Tuple of (labels, centers) where labels are cluster assignments and centers are centroids.
        """
        self.logger.debug(f"Starting K-Means with k={k}, max_iters={max_iters}")
        centers = x[np.random.choice(len(x), size=k, replace=False)]
        prev_centers = np.zeros_like(centers)
        labels = np.zeros(len(x))

        for iteration in range(max_iters):
            distances = np.linalg.norm(x[:, np.newaxis] - centers, axis=2)
            labels = np.argmin(distances, axis=1)
            prev_centers = centers.copy()
            for i in range(k):
                cluster_points = x[labels == i]
                if len(cluster_points) > 0:
                    centers[i] = np.mean(cluster_points, axis=0)
            if np.all(centers == prev_centers):
                self.logger.debug(f"K-Means converged after {iteration + 1} iterations")
                break
        return labels, centers

    def plot_animated_scatter(self, interval: int = 100, max_dots: int = 50) -> None:
        """Animate champion positions over time with a maximum of n dots per champion.
        
        Args:
            interval: Milliseconds between frames. Defaults to 100.
            max_dots: Maximum number of dots to show per champion. Defaults to 50.
        """
        map_height, map_width = self.background_image.shape[:2]
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(self.background_image, extent=[0, map_width, 0, map_height], alpha=0.6)

        # Sort data by timestamp
        self.df = self.df.sort_values("timestamp_dt")
        palette = sns.color_palette("hsv", len(self.champions))

        # Initialize empty scatter plots for each champion
        scatters = {}
        for i, champion in enumerate(self.champions):
            scatters[champion] = ax.scatter([], [], color=palette[i], alpha=0.7, s=20, label=champion)

        plt.title(f"Animated Champion Positions (Last {max_dots} Positions)", fontsize=14)
        plt.xlabel("X Coordinate (Scaled)")
        plt.ylabel("Y Coordinate (Scaled)")
        plt.xlim(0, map_width)
        plt.ylim(0, map_height)
        plt.legend(title="Champions", loc="upper right", fontsize=10)

        # Dictionary to store recent positions for each champion
        recent_positions = {champ: {'x': [], 'y': []} for champ in self.champions}

        def update(frame: int) -> list:
            """Update function for animation with max_dots limit."""
            current_row = self.df.iloc[frame]
            champion = current_row["champion"]
            
            # Add new position
            recent_positions[champion]['x'].append(current_row["x_scaled"])
            recent_positions[champion]['y'].append(current_row["y_scaled"])
            
            # Trim to only keep last max_dots positions
            if len(recent_positions[champion]['x']) > max_dots:
                recent_positions[champion]['x'] = recent_positions[champion]['x'][-max_dots:]
                recent_positions[champion]['y'] = recent_positions[champion]['y'][-max_dots:]
            
            # Update all scatters (not just the current champion)
            for champ in self.champions:
                if recent_positions[champ]['x']:  # Only update if we have data
                    scatters[champ].set_offsets(
                        np.c_[recent_positions[champ]['x'], recent_positions[champ]['y']]
                    )
            
            return list(scatters.values())

        anim = FuncAnimation(
            fig,
            update,
            frames=len(self.df),
            interval=interval,
            blit=True,
            repeat=False,
        )
        plt.tight_layout()
        self.logger.info(f"Starting animation with max {max_dots} dots per champion")
        plt.show()

    def plot_comparative_paths(self) -> None:
        """Visualize comparative movement patterns for all champions."""
        map_height, map_width = self.background_image.shape[:2]
        plt.figure(figsize=(8, 8))
        plt.imshow(self.background_image, extent=[0, map_width, 0, map_height], alpha=0.6)

        palette = sns.color_palette("hsv", len(self.champions))
        for i, champion in enumerate(self.champions):
            champ_data = self.df[self.df["champion"] == champion]
            plt.plot(
                champ_data["x_scaled"],
                champ_data["y_scaled"],
                marker="o",
                color=palette[i],
                alpha=0.7,
                linestyle="-",
                markersize=3,
                label=champion,
            )

        plt.title("Comparative Movement Patterns of All Champions", fontsize=14)
        plt.xlabel("X Coordinate (Scaled)")
        plt.ylabel("Y Coordinate (Scaled)")
        plt.xlim(0, map_width)
        plt.ylim(0, map_height)
        plt.legend(title="Champions", loc="upper right", fontsize=10)
        plt.tight_layout()
        self.logger.info("Generated comparative paths plot")
        plt.show()
    
    def plot_champion_scatter(self) -> None:
        """Visualize every data point of each champion as a scatter plot with color labels."""
        map_height, map_width = self.background_image.shape[:2]
        plt.figure(figsize=(8, 8))
        plt.imshow(self.background_image, extent=[0, map_width, 0, map_height], alpha=0.6)

        palette = sns.color_palette("hsv", len(self.champions))
        for i, champion in enumerate(self.champions):
            champ_data = self.df[self.df["champion"] == champion]
            plt.scatter(
                champ_data["x_scaled"],
                champ_data["y_scaled"],
                color=palette[i],
                alpha=0.7,
                s=20,  # Size of the dots
                label=champion,
                edgecolors="none",
            )

        plt.title("Champion Position Scatter Plot", fontsize=14)
        plt.xlabel("X Coordinate (Scaled)")
        plt.ylabel("Y Coordinate (Scaled)")
        plt.xlim(0, map_width)
        plt.ylim(0, map_height)
        plt.legend(title="Champions", loc="upper right", fontsize=10)
        plt.tight_layout()
        self.logger.info("Generated champion scatter plot")
        plt.show()

    def plot_heatmaps(self) -> None:
        """Generate color-coded heatmaps for each champion's position density."""
        map_height, map_width = self.background_image.shape[:2]
        plt.figure(figsize=(10, 10))
        plt.imshow(self.background_image, extent=[0, map_width, 0, map_height], alpha=0.6)
        
        palette = sns.color_palette("hsv", len(self.champions))
        
        # Create proxy artists for the legend
        legend_elements = []
        
        for i, champion in enumerate(self.champions):
            champ_data = self.df[self.df["champion"] == champion]
            sns.kdeplot(
                x=champ_data["x_scaled"],
                y=champ_data["y_scaled"],
                color=palette[i],
                alpha=0.5,
                fill=True,
                thresh=0.05,
                levels=10,
            )
            # Create a patch for the legend
            legend_elements.append(plt.Rectangle((0, 0), 1, 1, fc=palette[i], alpha=0.5, label=champion))
        
        plt.title("Champion Position Heatmaps", fontsize=14)
        plt.xlabel("X Coordinate (Scaled)")
        plt.ylabel("Y Coordinate (Scaled)")
        plt.xlim(0, map_width)
        plt.ylim(0, map_height)
        plt.legend(handles=legend_elements, title="Champions", loc="upper right", fontsize=10)
        plt.tight_layout()
        self.logger.info("Generated champion heatmaps")
        plt.show()

if __name__ == "__main__":
    plotter = Plotter()
    plotter.download_minimap()
    plotter.load_and_scale_data()
    #plotter.plot_comparative_paths()
    #plotter.plot_champion_scatter()
    #plotter.plot_heatmaps()
    plotter.plot_animated_scatter(max_dots=10)
