# Spectral Matching Demo
# Author: Kevin Bedros
# Date: 2/8/2026

# Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Extracting X and Y from CSV files
def load_csv(path):

    # Reading file
    df = pd.read_csv(path)

    # Assigning variables for plotting and creating arrays
    x = df[df.columns[0]].to_numpy(dtype=float)
    y = df[df.columns[1]].to_numpy(dtype=float)

    # Checks if X and Y are both real and removes any NaN
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]

    # Sprts by X values
    order = np.argsort(x)
    x, y = x[order], y[order]

    # Remove duplicates
    x_place, y_place = np.unique(x, return_index=True)
    x, y = x_place, y[y_place]

    return x, y

# Resampling to a common grid
def create_grid(x1, y1, x2, y2):

    # Find min and max wavelength
    min_v = max(x1.min(), x2.min())
    max_v = min(x1.max(), x2.max())

    # Check if spectra overlap
    if max_v <= min_v:
        raise ValueError("Spectra do not overlap in wavelength range.")

    # Determine step size
    s1 = np.median(np.diff(x1))
    s2 = np.median(np.diff(x2))
    step = max(s1, s2)

    # Create common X-axis
    grid = np.arange(min_v, max_v + 0.5 * step, step)

    # New Y values
    new_y1 = np.interp(grid, x1, y1)
    new_y2 = np.interp(grid, x2, y2)

    return grid, new_y1, new_y2

# Spectral Angle Mapper (SAM) calculation
def spectral_angle_mapper(y1, y2, eps=1e-12):

    # Ensure inputs are numpy arrays
    y1 = np.asarray(y1, dtype=float)
    y2 = np.asarray(y2, dtype=float)

    # Calculate dot product and norms
    dot = float(np.dot(y1, y2))
    n1 = float(np.linalg.norm(y1))
    n2 = float(np.linalg.norm(y2))

    cos_sim = np.clip(dot / max(n1 * n2, eps), -1.0, 1.0)

    # Calculate angle in radians and degrees
    angle_rad = float(np.arccos(cos_sim))
    angle_deg = float(np.degrees(angle_rad))
    
    # Normalized SAM (0 to 1)
    sam_norm = 1.0 - (2.0 / np.pi) * angle_rad
    sam_norm = float(np.clip(sam_norm, 0.0, 1.0))


    return angle_rad, angle_deg, float(cos_sim), sam_norm

if __name__ == "__main__":

    # What are the user files
    file1 = "extracted_photodiode_spectrum.csv"
    file2 = "extracted_photoluminescence_spectrum.csv"

    # Pull X and Y 
    x1, y1 = load_csv(file1)
    x2, y2 = load_csv(file2)

    # Create a new common grid with interpolated Y values
    grid, new_y1, new_y2 = create_grid(x1, y1, x2, y2)


    angle_rad, angle_deg, cos_sim, sam_norm = spectral_angle_mapper(new_y1, new_y2)

    print(f"Common grid points: {len(grid)}")
    print(f"SAM angle: {angle_rad:.6f} rad  ({angle_deg:.4f} deg)")
    print(f"Cosine similarity: {cos_sim:.6f}")
    print(f"Normalized SAM: {sam_norm:.6f}")

    # Plotting the new spectrum
    plt.plot(grid, new_y1, label="Photodiode Spectrum (Original)", linestyle="--")
    plt.plot(grid, new_y2, label="Photoluminescence Spectrum (Reference)", linewidth=2)
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Intensity")
    plt.title("Spectral Matching")
    plt.legend()
    plt.show()

