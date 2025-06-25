"""Animated scatter plot with predicted champion positions."""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import seaborn as sns
from pathlib import Path
from utils import setup_logger

class ChampionPlotter:
    """Handles animated visualization of champion positions with predictions."""

    def __init__(self, timeline_path: str = "data/timeline.csv", minimap_path: str = "data/minimap.png"):
        """Initialize with paths to timeline CSV and minimap image."""
        self.logger = setup_logger("champion_plotter")
        self.project_root = Path(__file__).parent
        self.timeline_path = self.project_root / timeline_path
        self.minimap_path = self.project_root / minimap_path
        self.df = None
        self.champions = None
        self.median_velocities = None
        self.background_image = plt.imread(self.minimap_path)
        self.load_and_scale_data()

    def load_and_scale_data(self) -> None:
        """Load timeline data, compute velocities, and predict positions."""
        self.logger.info(f"Loading timeline from {self.timeline_path}")
        self.df = pd.read_csv(self.timeline_path)
        
        # Convert timestamps and sort
        self.df["timestamp_dt"] = pd.to_datetime(self.df["timestamp"])
        self.df = self.df.sort_values("timestamp_dt")
        
        # Scale X and Y to minimap dimensions
        map_height, map_width = self.background_image.shape[:2]
        captured_width, captured_height = 252, 252  # From your config defaults
        self.df["x_scaled"] = (self.df["X"] / captured_width) * map_width
        self.df["y_scaled"] = ((captured_height - self.df["Y"]) / captured_height) * map_height
        
        self.champions = self.df["champion"].unique()
        self.logger.info(f"Loaded data for champions: {list(self.champions)}")
        
        # Compute median velocities
        self.median_velocities = self.compute_median_velocities()
        self.logger.debug(f"Median velocities: {self.median_velocities}")
        
        # Interpolate missing positions
        self.interpolate_positions()

    def compute_median_velocities(self) -> dict:
        """Compute median velocity (vx, vy) for each champion."""
        median_velocities = {}
        for champion in self.champions:
            champ_data = self.df[self.df["champion"] == champion]
            if len(champ_data) < 2:
                median_velocities[champion] = (0.0, 0.0)
                continue
            dt = champ_data["timestamp_dt"].diff().dt.total_seconds().iloc[1:]
            dx = champ_data["x_scaled"].diff().iloc[1:]
            dy = champ_data["y_scaled"].diff().iloc[1:]
            vx = dx / dt
            vy = dy / dt
            median_velocities[champion] = (np.nanmedian(vx), np.nanmedian(vy))
        return median_velocities

    def interpolate_positions(self) -> None:
        """Interpolate positions between timestamps using median velocity."""
        self.df["x_predicted"] = self.df["x_scaled"]
        self.df["y_predicted"] = self.df["y_scaled"]
        
        # Create a full time series with 1-second intervals
        start_time = self.df["timestamp_dt"].min()
        end_time = self.df["timestamp_dt"].max()
        all_times = pd.date_range(start=start_time, end=end_time, freq="1s")  # Fixed 'S' to 's'
        full_df = pd.DataFrame({"timestamp_dt": all_times})
        
        for champion in self.champions:
            champ_df = self.df[self.df["champion"] == champion][["timestamp_dt", "x_scaled", "y_scaled"]]
            merged_df = full_df.merge(champ_df, on="timestamp_dt", how="left")
            last_x, last_y = np.nan, np.nan
            last_time = None
            vx, vy = self.median_velocities[champion]
            
            for idx, row in merged_df.iterrows():
                if not pd.isna(row["x_scaled"]):
                    last_x, last_y = row["x_scaled"], row["y_scaled"]
                    last_time = row["timestamp_dt"]
                    merged_df.at[idx, "x_predicted"] = last_x
                    merged_df.at[idx, "y_predicted"] = last_y
                elif last_x is not np.nan:
                    time_delta = (row["timestamp_dt"] - last_time).total_seconds()
                    merged_df.at[idx, "x_predicted"] = last_x + vx * time_delta
                    merged_df.at[idx, "y_predicted"] = last_y + vy * time_delta
            
            # Merge back into main DataFrame
            full_df[f"{champion}_x"] = merged_df["x_predicted"]
            full_df[f"{champion}_y"] = merged_df["y_predicted"]
        
        self.full_df = full_df
        self.logger.info("Interpolated positions for animation")

    def plot_animated_scatter(self, interval: int = 100) -> None:
        """Animate champion positions with predictions."""
        map_height, map_width = self.background_image.shape[:2]
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(self.background_image, extent=[0, map_width, 0, map_height], alpha=0.6)

        # Initialize scatter plots
        scatters = {}
        palette = sns.color_palette("hsv", len(self.champions))
        for i, champion in enumerate(self.champions):
            scatters[champion] = ax.scatter([], [], color=palette[i], alpha=0.7, s=30, label=champion)

        plt.title("Animated Champion Positions with Predictions", fontsize=14)
        plt.xlabel("X Coordinate (Scaled)")
        plt.ylabel("Y Coordinate (Scaled)")
        plt.xlim(0, map_width)
        plt.ylim(0, map_height)
        plt.legend(loc="upper right", fontsize=8)

        def update(frame: int) -> list:
            """Update scatter positions for each frame."""
            frame_data = self.full_df.iloc[frame]
            scatter_list = []
            for champion in self.champions:
                x = frame_data[f"{champion}_x"]
                y = frame_data[f"{champion}_y"]
                if not pd.isna(x) and not pd.isna(y):
                    scatters[champion].set_offsets([[x, y]])
                else:
                    scatters[champion].set_offsets(np.empty((0, 2)))  # Fixed empty offsets
                scatter_list.append(scatters[champion])
            return scatter_list

        anim = FuncAnimation(fig, update, frames=len(self.full_df), interval=interval, 
                            blit=True, repeat=False)
        plt.tight_layout()
        self.logger.info("Starting animation. Close the plot window to continue.")
        plt.show()

if __name__ == "__main__":
    plotter = ChampionPlotter()
    plotter.plot_animated_scatter(interval=100)