import pandas as pd
from scipy.stats import boxcox
from scipy import stats
from cycler import cycler
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from scipy.interpolate import griddata
import re


def calc_mean_ci_for_encoding(df, start_col, end_col, confidence=0.95):
    results = []

    for encoding in df['enc'].unique():
        subset = df[df['enc'] == encoding]
        scores = subset.iloc[:, start_col:end_col].values
        flattened_scores = scores.flatten()
        flattened_scores = flattened_scores[~np.isnan(flattened_scores)]
        mean_score = np.mean(flattened_scores)
        sem = stats.sem(flattened_scores)
        ci = stats.t.interval(confidence, len(flattened_scores) - 1, loc=mean_score, scale=sem)

        results.append({
            'encoding': encoding,
            'mean': mean_score,
            'ci_lower': ci[0],
            'ci_upper': ci[1]
        })

    results_df = pd.DataFrame(results)

    encodings = results_df['encoding']
    means = results_df['mean']
    ci_lowers = means - results_df['ci_lower']
    ci_uppers = results_df['ci_upper'] - means

    return encodings, means, ci_lowers, ci_uppers


def extract_encoding(filename):

    match = re.search(r'(F|H)OA_\d+k', filename)
    encoding = match.group(0)

    return encoding


def boxcox_transform(values, normalise=False, eps=1e-6):
    if normalise:
        min_val = np.nanmin(values)
        max_val = np.nanmax(values)
        values = np.clip((values - min_val) / (max_val - min_val + eps), eps, None)  # Ensure values are > 0

    # Ensure all values are strictly positive by shifting if necessary
    min_val = np.min(values)
    if min_val <= 0:
        values = values + abs(min_val) + eps  # Shift values to be > 0

    transformed_values, lmbda = boxcox(values)

    # Rescale the transformed values to [0, 1]
    transformed_min = np.nanmin(transformed_values)
    transformed_max = np.nanmax(transformed_values)
    normalized_transformed_values = (transformed_values - transformed_min) / (transformed_max - transformed_min + eps)

    return normalized_transformed_values


# Function to calculate mean, lower, and upper confidence intervals
def calculate_ci(data, confidence=0.95):
    mean = np.mean(data)
    sem = stats.sem(data)  # Standard error of the mean
    margin = sem * stats.t.ppf((1 + confidence) / 2.0, len(data) - 1)  # Margin of error
    return mean, mean - margin, mean + margin


def plot_half_sphere(azimuths_deg, elevations_deg, heat_map_values, plot_title):
    # Normalize azimuthal range to [-180, 180]
    azimuths_deg = np.mod(np.array(azimuths_deg) + 180, 360) - 180

    # Find the index of the first -180
    index = np.where(azimuths_deg == -180)[0][0]

    # Get the number of unique elevations for each azimuth
    elevations_num = len(set(elevations_deg))

    # Insert elevations_num 180s before the first -180, to ensure symmetry
    azimuths_deg = np.insert(azimuths_deg, index, [180] * elevations_num)

    elevations = [0, 5, 10, 15, 20, 25, 30, 40, 50, 60, 75, 90,
                  -5, -10, -15, -20, -25, -30, -40, -50, -60, -75, -90]

    # Insert elevations in elevations_deg
    elevations_deg = np.insert(elevations_deg, index, elevations)

    # Copy and insert heat map values for the number of elevations
    values_to_copy = heat_map_values[index:index + elevations_num]
    heat_map_values = np.insert(heat_map_values, index, values_to_copy)

    # Convert degrees to radians
    azimuths = np.radians(azimuths_deg)
    elevations = np.radians(elevations_deg)

    # Convert spherical coordinates to Cartesian coordinates (Longitude, Latitude)
    longitudes = np.degrees(azimuths)
    latitudes = np.degrees(elevations)

    # Define the grid for interpolation
    num_longitudes = 180
    num_latitudes = 180
    longitude_grid = np.linspace(0, 180, num_longitudes)
    latitude_grid = np.linspace(-90, 90, num_latitudes)
    longitude_grid, latitude_grid = np.meshgrid(longitude_grid, latitude_grid)

    # Interpolate the heat map values onto the grid
    interpolated_values = griddata((longitudes, latitudes), heat_map_values,
                                   (longitude_grid, latitude_grid), method='linear')

    # Plotting
    fig = plt.figure(figsize=(11, 8.5))
    projMoll = ccrs.Robinson(central_longitude=0)
    ax = plt.subplot(1, 1, 1, projection=projMoll)

    # Add title
    ax.set_title(plot_title, fontsize=22)

    # Customize grid labels
    gl = ax.gridlines(draw_labels=True, color='gray', linestyle='--', alpha=0.7)
    gl.right_labels = False
    gl.top_labels = False

    # Custom formatter for longitude labels with degree sign and +/- notation
    def custom_longitude_formatter(value):
        if value == 0:
            return "0°"
        elif value > 0:
            return f"+{int(value)}°"
        else:
            return f"{int(value)}°"

    # Custom formatter for latitude labels with degree sign
    def custom_latitude_formatter(value):
        return f"{int(value)}°"

    # Increase font size for labels
    gl.xlabel_style = {'size': 22}
    gl.ylabel_style = {'size': 22}

    # Apply custom formatters
    gl.xformatter = plt.FuncFormatter(lambda x, _: custom_longitude_formatter(x))
    gl.yformatter = plt.FuncFormatter(lambda y, _: custom_latitude_formatter(y))

    # Limit the extent to the right half of the sphere
    ax.set_extent([0, 180, -90, 90], crs=ccrs.PlateCarree())

    # Plot the interpolated heat map
    heatmap = ax.pcolormesh(longitude_grid, latitude_grid, interpolated_values,
                            transform=ccrs.PlateCarree(), cmap='Greens')

    # Add color bar
    cbar = plt.colorbar(heatmap, orientation='horizontal', pad=0.05)
    cbar.set_label('LS', fontsize=22)
    cbar.ax.tick_params(labelsize=22)

    plt.savefig("./plots/localization_sensitivity_sph_" + plot_title + ".png")

    # Show the plot
    plt.show()


def loc_sensitivity_sphere_plots(results_path):
    results_df = pd.read_csv(results_path)

    sources = results_df["source"].unique()

    # Iterate over each source and plot in the corresponding subplot
    for i, source in enumerate(sources):  # Limit to 8 sources
        # Get data for the current source
        source_values = results_df[results_df["source"] == source]

        azimuths_deg = source_values.az.tolist()
        elevations_deg = source_values.el.tolist()
        ls_lst = source_values.LS.tolist()

        ## apply box-cox transform
        ls_lst = boxcox_transform(ls_lst, normalise=False)

        plot_half_sphere(azimuths_deg, elevations_deg, ls_lst, plot_title=source)


def loc_sensitivity_line_plots(results_path):

    df = pd.read_csv(results_path)

    # Convert az from [0, 360] to [-180, 180]
    df["az"] = ((df["az"] + 180) % 360) - 180

    # Define valid azimuth and elevation values
    valid_az = {0, 30, 60, 92, 120, 156, 180, -30, -60, -88, -120, -156, -180}
    valid_el = {0, 30, 60, 90, -30, -60, -90}

    # Filter rows
    df_filtered = df[df["az"].isin(valid_az) & df["el"].isin(valid_el)]

    # Duplicate rows where az = 180 and change az to -180
    df_180 = df_filtered[df_filtered["az"] == -180].copy()
    df_180["az"] = 180

    # Append duplicated rows to the filtered DataFrame
    df = pd.concat([df_filtered, df_180], ignore_index=True)

    df = df[df["source"] == "Female Reverb"]

    # Ensure correct data types
    df["az"] = df["az"].astype(float)
    df["el"] = df["el"].astype(float)
    df["LS"] = df["LS"].astype(float)

    # Sort by azimuth values (smallest to largest)
    df = df.sort_values(by="az")

    # Define line styles for different elevations
    line_styles = ["k--", "b--", "g--", "r-", "g:", "b:", "k:"]  # Matching the style in the image

    # Unique elevation values
    unique_elevations = sorted(df["el"].unique(), reverse=True)

    # Create the plot
    plt.figure(figsize=(8, 5))
    for i, el in enumerate(unique_elevations):
        subset = df[df["el"] == el]  # Filter data for a specific elevation
        plt.plot(subset["az"], subset["LS"], line_styles[i % len(line_styles)], label=f"El={int(el)}°")

    # Formatting
    plt.title("Female rev., Reference Az=0°,El=0°", fontsize=16)
    plt.xlabel("Azimuth (°)", fontsize=16)
    plt.ylabel("LS", fontsize=16)
    plt.legend(loc="upper left", fontsize=14)
    plt.xticks(range(-180, 181, 30), fontsize=12)  # Set x-axis ticks from -180 to 180
    plt.yticks(fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.xlim(-180, 180)
    plt.ylim(0, 1)

    plt.savefig("./plots/localization_sensitivity_line_female_rev.png")

    # Show plot
    plt.show()


def interpolation_plots(results_path):
    results_df = pd.read_csv(results_path)

    results_df["angles"] = ("(" + results_df["az1"].astype(str) + "," + results_df["el1"].astype(str) + "), \n" +
                            "(" + results_df["az2"].astype(str) + "," + results_df["el2"].astype(str) + ")")

    # Separate the data
    az_50_df = results_df[
        (results_df["az1"] + results_df["az2"] == 100) | ((results_df["az1"] == 50) & (results_df["az2"] == np.nan))]
    az_90_df = results_df[(results_df["az1"] + results_df["az2"] == 180) | (results_df["az1"] == 90)]

    plt.figure(figsize=(12, 10))

    plt.rc('axes', prop_cycle=(cycler('marker', ['o', 's', 'D', '<', '>', 'p', '*', 'h', '+', 'x']) +
                               cycler('color', plt.cm.tab10.colors)))  # Also cycle through colors

    for source, group_data in az_50_df.groupby("source"):
        plt.plot(group_data["angles"], group_data["LS"], label=source, markersize=10)  # Plot each group

    # Add labels, legend, and title
    plt.xlabel("Interpolation Angle Pairs", fontsize=20)
    plt.ylabel("LS", fontsize=20)
    plt.title("Reference angle (Az, El): (50, 0)", fontsize=20)
    plt.legend(loc="upper right", fontsize=14)

    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)

    plt.savefig("./plots/interpolation_az_50.png")

    # Show the plot
    plt.show()

    plt.figure(figsize=(12, 10))

    plt.rc('axes', prop_cycle=(cycler('marker', ['o', 's', 'D', '<', '>', 'p', '*', 'h', '+', 'x']) +
                               cycler('color', plt.cm.tab10.colors)))  # Also cycle through colors

    for source, group_data in az_90_df.groupby("source"):
        plt.plot(group_data["angles"], group_data["LS"], label=source, markersize=10)  # Plot each group

    # Add labels, legend, and title
    plt.xlabel("Interpolation Angle Pairs", fontsize=20)
    plt.ylabel("LS", fontsize=20)
    plt.title("Reference angle (Az, El): (90, 0)", fontsize=20)
    plt.legend(loc="upper right", fontsize=14)

    #plt.xticks(rotation=45, fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)

    plt.savefig("./plots/interpolation_az_90.png")

    # Show the plot
    plt.show()


def spleaker_layouts_plots(results_path):
    results_df = pd.read_csv(results_path)

    results_df = results_df[results_df["speaker_layout"] != "none"]

    # Separate the data
    has_not_el_df = results_df[results_df["el1"] == 0]
    has_el_df = results_df[results_df["el1"] != 0]

    # Aggregate with confidence intervals for has_not_el_df
    has_not_el_ci = has_not_el_df.groupby("speaker_layout")["LS"].apply(
        lambda x: calculate_ci(x)
    )
    has_not_el_df_agg = pd.DataFrame({
        "speaker_layout": has_not_el_ci.index,
        "mean_LS": has_not_el_ci.apply(lambda x: x[0]),
        "lower_ci": has_not_el_ci.apply(lambda x: x[1]),
        "upper_ci": has_not_el_ci.apply(lambda x: x[2]),
    })

    # Plot the first bar chart
    plt.figure(figsize=(8, 6))
    plt.rcParams.update({'xtick.labelsize': 14, 'ytick.labelsize': 14})
    plt.bar(has_not_el_df_agg["speaker_layout"], has_not_el_df_agg["mean_LS"], color='grey',
            width=0.3, capsize=5, edgecolor='black',
            yerr=[
                has_not_el_df_agg["mean_LS"] - has_not_el_df_agg["lower_ci"],
                has_not_el_df_agg["upper_ci"] - has_not_el_df_agg["mean_LS"]
            ])

    # Add labels and title for the first plot
    plt.xlabel("Speaker Layout", fontsize=15)
    plt.ylabel("LS", fontsize=15)

    plt.title("Angles (Az,El) = (60,0), (300,0)", fontsize=15)

    plt.savefig("./plots/speaker_layouts_without_el.png")

    # Show the first plot
    plt.show()

    # Aggregate with confidence intervals for has_el_df
    has_el_ci = has_el_df.groupby("speaker_layout")["LS"].apply(
        lambda x: calculate_ci(x)
    )
    has_el_df_agg = pd.DataFrame({
        "speaker_layout": has_el_ci.index,
        "mean_LS": has_el_ci.apply(lambda x: x[0]),
        "lower_ci": has_el_ci.apply(lambda x: x[1]),
        "upper_ci": has_el_ci.apply(lambda x: x[2]),
    })

    # Plot the second bar chart
    plt.figure(figsize=(8, 6))
    plt.rcParams.update({'xtick.labelsize': 14, 'ytick.labelsize': 14})
    plt.bar(has_el_df_agg["speaker_layout"], has_el_df_agg["mean_LS"], color='grey',
            width=0.3, capsize=5, edgecolor='black',
            yerr=[
                has_el_df_agg["mean_LS"] - has_el_df_agg["lower_ci"],
                has_el_df_agg["upper_ci"] - has_el_df_agg["mean_LS"]
            ])

    # Add labels and title for the second plot
    plt.xlabel("Speaker Layout", fontsize=15)
    plt.ylabel("LS", fontsize=15)
    plt.title("Angles (Az,El) = (60,50), (300,30)", fontsize=15)

    plt.savefig("./plots/speaker_layouts_with_el.png")

    # Show the second plot
    plt.show()


def compression_plots(results_single_source_path, results_multi_source_path):
    # Load the data for experiment 1
    single_df = pd.read_csv(results_single_source_path)

    # Load the data for experiment 2
    multi_df = pd.read_csv(results_multi_source_path)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(32, 15))

    # plot 1
    single_df['enc'] = single_df['test'].apply(extract_encoding)
    encodings, means, ci_lowers, ci_uppers = calc_mean_ci_for_encoding(single_df, 4, 5)

    bar_width = 0.2
    ax1.bar(encodings, means, color='grey', width=bar_width, yerr=[ci_lowers, ci_uppers], capsize=5,
            edgecolor='black')
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.tick_params(labelsize=20)
    ax1.tick_params(axis='x', rotation=45)
    ax1.set_xlabel('Encoding', fontsize=20)
    ax1.set_ylabel('LS', fontsize=20)
    ax1.set_title('BINAQUAL LS (single source)', fontsize=20, fontweight='bold')

    # pLot 2
    multi_df['enc'] = multi_df['test'].apply(extract_encoding)
    encodings, means, ci_lowers, ci_uppers = calc_mean_ci_for_encoding(multi_df, 4, 5)

    bar_width = 0.2
    ax2.bar(encodings, means, color='grey', width=bar_width, yerr=[ci_lowers, ci_uppers], capsize=5,
            edgecolor='black')
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.tick_params(labelsize=20)
    ax2.tick_params(axis='x', rotation=45)
    ax2.set_xlabel('Encoding', fontsize=20)
    ax2.set_ylabel('LS', fontsize=20)
    ax2.set_title('BINAQUAL LS (multiple source)', fontsize=20, fontweight='bold')

    fig.savefig("./plots/compression_test.png", bbox_inches='tight')

    plt.tight_layout()
    plt.show()



if __name__ == '__main__':

    result_path = "./results/loc_sensitivity_test.csv"
    loc_sensitivity_sphere_plots(result_path)
    loc_sensitivity_line_plots(result_path)

    result_path = "./results/interpolation_test.csv"
    interpolation_plots(result_path)

    result_path = "./results/speakers_layouts_test.csv"
    spleaker_layouts_plots(result_path)

    results_single_source_path = "./results/compression_single_source_test.csv"
    results_multi_source_path = "./results/compression_multi_source_test.csv"
    compression_plots(results_single_source_path, results_multi_source_path)