from binaqual import calculate_binaqual
import pandas as pd


def _process_and_save_results(inputs_df, output_path, audio_path, get_ref_func):
    """
    Helper function to compute localization similarity and save the results.

    Args:
        inputs_df (pd.DataFrame): Input metadata dataframe.
        output_path (str): Path to save the resulting CSV file.
        audio_path (str): Directory containing audio files.
        get_ref_func (Callable): Function to generate reference filename for each row.

    Returns:
        None
    """
    nsim_1_lst, nsim_2_lst, LS_lst, ref_lst = [], [], [], []

    for i, row in inputs_df.iterrows():
        ref = get_ref_func(row)
        test = row["file_name"]

        nsims, LS = calculate_binaqual(
            audio_path + ref,
            audio_path + test,
        )
        nsim_1_lst.append(nsims[0])
        nsim_2_lst.append(nsims[1])
        LS_lst.append(LS)
        ref_lst.append(ref)

    inputs_df["nsim_1"] = nsim_1_lst
    inputs_df["nsim_2"] = nsim_2_lst
    inputs_df["LS"] = LS_lst
    inputs_df["ref"] = ref_lst
    inputs_df = inputs_df.rename(columns={'file_name': 'test'})
    inputs_df = inputs_df[["ref"] + [col for col in inputs_df.columns if col != "ref"]]
    inputs_df.to_csv(output_path, index=False)


def loc_sensitivity_test(metadata_path, audio_path, output_path, ref_angle,
                         subject_id, ir_type):
    """
    Localization sensitivity test

    Args:
        metadata_path (str): CSV file path with metadata.
        audio_path (str): Path to audio files.
        output_path (str): Path to save the output CSV.
        ref_angle (tuple): Tuple of (azimuth, elevation) for reference.
        subject_id (str): Subject identifier.
        ir_type (str): Type of impulse response.

    Returns:
        None
    """
    inputs_df = pd.read_csv(metadata_path)

    def get_ref(row):
        return f"{row['source']}_{subject_id}_{ir_type}_az_{ref_angle[0]}_el_{ref_angle[1]}.wav"

    _process_and_save_results(inputs_df, output_path, audio_path, get_ref)


def interpolation_test(metadata_path, audio_path, output_path, subject_id, ir_type):
    """
    Interpolation test

    Args:
        metadata_path (str): Path to metadata CSV.
        audio_path (str): Path to audio files.
        output_path (str): CSV save path.
        subject_id (str): Subject identifier.
        ir_type (str): Type of impulse response.

    Returns:
        None
    """
    inputs_df = pd.read_csv(metadata_path)

    def get_ref(row):
        az1, az2 = row["az1"], row["az2"]
        ref_az = az1 if pd.isna(az2) else int((az1 + az2) / 2)
        return f"{row['source']}_{subject_id}_{ir_type}_az_{ref_az}_el_0.wav"

    _process_and_save_results(inputs_df, output_path, audio_path, get_ref)


def speaker_layouts_test(metadata_path, audio_path, output_path):
    """
    Speaker layout test

    Args:
        metadata_path (str): Path to metadata CSV.
        audio_path (str): Directory with audio files.
        output_path (str): Output CSV path.

    Returns:
        None
    """
    inputs_df = pd.read_csv(metadata_path)

    def get_ref(row):
        test = row["file_name"]
        layout = test.split("_")[-1]
        return test.replace(layout, "none.wav") if layout != "none.wav" else test

    _process_and_save_results(inputs_df, output_path, audio_path, get_ref)


def compression_test(metadata_path, audio_path, output_path):
    """
    Compression test

    Args:
        metadata_path (str): CSV metadata file.
        audio_path (str): Audio file directory.
        output_path (str): Output CSV path.

    Returns:
        None
    """
    inputs_df = pd.read_csv(metadata_path)

        def get_ref(row):
        test = row["file_name"]
        bitrate = test.split("_")[-2]
        enc = test.split("_")[-3]
        if enc == "FOA":
            test = test.replace("FOA", "HOA")
        return test.replace(bitrate, "REF")

    _process_and_save_results(inputs_df, output_path, audio_path, get_ref)


if __name__ == '__main__':

    # reference angle for localization sensitivity test
    ref_angle = (0, 0)

    subject_id = "D2"
    ir_type = "HRIR"

    metadata_path = "../SynBAD/localization_sensitivity/metadata.csv"
    audio_path = "../SynBAD/localization_sensitivity/"
    output_path = "./results/loc_sensitivity_test.csv"
    loc_sensitivity_test(metadata_path, audio_path, output_path, ref_angle, subject_id, ir_type)

    metadata_path = "../SynBAD/angle_interpolations/metadata.csv"
    audio_path = "../SynBAD/angle_interpolations/"
    output_path = "./results/interpolation_test.csv"
    interpolation_test(metadata_path, audio_path, output_path, subject_id, ir_type)

    metadata_path = "../SynBAD/speaker_layouts/metadata.csv"
    audio_path = "../SynBAD/speaker_layouts/"
    output_path = "./results/speakers_layouts_test.csv"
    speaker_layouts_test(metadata_path, audio_path, output_path)

    metadata_path = "../SynBAD/codec_compression/metadata_single_source.csv"
    audio_path = "../SynBAD/codec_compression/"
    output_path = "./results/compression_single_source_test.csv"
    compression_test(metadata_path, audio_path, output_path)

    metadata_path = "../SynBAD/codec_compression/metadata_multi_source.csv"
    audio_path = "../SynBAD/codec_compression/"
    output_path = f"./results/compression_multi_source_test.csv"
    compression_test(metadata_path, audio_path, output_path)