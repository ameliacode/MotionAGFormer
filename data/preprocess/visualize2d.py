import glob

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

# Set your path here
data_path = "./data/keypoints"

# Find all _2D.npy files
files = glob.glob(f"{data_path}/*_2D.npy")
print(f"Found {len(files)} files")

connections = [
    (10, 9),
    (9, 8),
    (8, 7),
    (8, 14),
    (8, 11),
    (14, 15),
    (15, 16),
    (11, 12),
    (12, 13),
    (0, 7),
    (0, 1),
    (0, 4),
    (1, 2),
    (2, 3),
    (4, 5),
    (5, 6),
]


def show_animation():
    """Show each file separately, one after another"""

    for i, file in enumerate(files):
        print(f"\nShowing file {i+1}/{len(files)}: {file.split('/')[-1]}")

        data = np.load(file)
        print(f"Shape: {data.shape}")

        fig, ax = plt.subplots(figsize=(8, 8))

        def animate(frame):
            ax.clear()
            pose = data[frame]

            # Draw skeleton bones
            for start, end in connections:
                start_joint = pose[start][..., :2]
                end_joint = pose[end][..., :2]

                if (
                    not (start_joint == [0, 0]).all()
                    and not (end_joint == [0, 0]).all()
                ):
                    ax.plot(
                        [start_joint[0], end_joint[0]],
                        [start_joint[1], end_joint[1]],
                        "b-",
                        linewidth=2,
                    )

            for j, joint in enumerate(pose[..., :2]):
                if not (joint == [0, 0]).all():
                    ax.plot(joint[0], joint[1], "ro", markersize=5)
                    ax.text(
                        joint[0],
                        joint[1] - 10,
                        str(j),
                        fontsize=10,
                        color="blue",
                        ha="center",
                        va="bottom",
                        weight="bold",
                    )

            ax.set_aspect("equal")
            ax.invert_yaxis()
            ax.set_title(f"{file.split('/')[-1]} - Frame {frame+1}/{len(data)}")

        anim = animation.FuncAnimation(
            fig, animate, frames=len(data), interval=100, repeat=True
        )
        plt.show()

        # Wait for user to close window before showing next file
        input("Press Enter to continue to next file (or Ctrl+C to exit)...")


# Run this:
if __name__ == "__main__":
    show_animation()
