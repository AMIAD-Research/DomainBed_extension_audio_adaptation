"""
Generation of an augmented version of a speech corpus: generation of the annotation table.
"""

import argparse
import numpy as np
import os
import pandas as pd
import random as rd 



def get_subpath_from_folder(full_path, folder_name="wham_noise"):
    """
    Returns the part of full_path starting from folder_name.
    Example:
        /data/datasets/wham_noise/tr/file.wav
        -> wham_noise/tr/file.wav
    """
    if full_path == None: 
        return None
    else: 
        # Normalize separators for cross-platform
        parts = os.path.normpath(full_path).split(os.sep)
        if folder_name in parts:
            idx = parts.index(folder_name)
            return os.path.join(*parts[idx:])
        else:
            # fallback: just return the filename
            return os.path.basename(full_path)

def uniform_distribution(min=-20, max=20): 
    return rd.uniform(min, max)


def get_snr_sampler(
    distribution,
    snr_range,
):
    """
    distribution: ["uniform", "triangular"]
    returns: callable snr_sampler()
    """
    if distribution == "uniform":
        return lambda: uniform_distribution(snr_range[0], snr_range[2])
    else: 
        raise ValueError(f"Unkown distribution {distribution}")

def get_gain_sampler(
    distribution,
    gain_range,
):
    """
    distribution: ["uniform", "triangular"]
    returns: callable gain_sampler()
    """
    if distribution == "uniform":
        return lambda: uniform_distribution(gain_range[0], gain_range[2])
    else: 
        raise ValueError(f"Unkown distribution {distribution}")

def parse_args():

    """ parse_args """

    parser = argparse.ArgumentParser(
        description='Augmentation')
    
    parser.add_argument(
        "--input_rootdir", 
        type=str,
        help="input rootdir"
    )

    parser.add_argument(
            "--output_rootdir", 
            type=str,
            help="output rootdir"
        )

    parser.add_argument(
        "--input_list", 
        type=str,
        help="input_list"
    )

    parser.add_argument(
        "--augmentation", 
        type=str,
        help="augmentation method",
        default=""
    )

    parser.add_argument(
            "--wham", 
            type=str,
            help="wham_path",
            default=""
        )
    parser.add_argument("--snr_distribution", type=str, default="triangular", choices=["uniform", "triangular"], help="SNR distribution")

    parser.add_argument("--wind_snr_range", type=str, default=[-40, -30, -20], help="SNR range for sampler - wind noise") 
    parser.add_argument("--wham_snr_range", type=str, default=[-15, -5, 5], help="SNR range for sampler - wind noise") 
    parser.add_argument("--saturation_gain_range", type=str, default=[0.5, 2.5, 4.5], help="gain range for sampler - saturation noise") 



    return parser.parse_args()


if __name__=="__main__":
    args = vars(parse_args())
    input_rootdir = args["input_rootdir"]
    input_list = args["input_list"]
    output_rootdir = args["output_rootdir"]
    augmentation = args["augmentation"]
    wham = args["wham"]

    os.makedirs(output_rootdir, exist_ok=True)

    # read list and create annotation table

    table_list = pd.read_csv(input_list)
    if not {"annotation_file","id"}.issubset(table_list.columns):
        raise ValueError(f"The list file {input_list} should contain at least 'annotation_file' and 'id' columns")

    annotation_filenames = list(set(table_list['annotation_file'].values.tolist()))
    table_annotation = None
    for annotation_filename in annotation_filenames:
        subset = table_list[table_list['annotation_file'].isin([annotation_filename])]
        ids = subset['id'].values.tolist()

        filePath = os.path.join(input_rootdir,annotation_filename)
        print(filePath)
        df = pd.read_csv(filePath)
        if not {"id"}.issubset(table_list.columns):
            raise ValueError(f"The list file {filePath} should contain at least the 'id' column")      
        df = df[df['id'].isin(ids)]

        if table_annotation is None:
            table_annotation = df
        else:
            if df.columns.difference(table_annotation.columns).empty:
                table_annotation = pd.concat([table_annotation,df],ignore_index=True)
            else:
                raise ValueError(f"The annotation file {filePath} should contain the same columns as the previous ones")
    table_annotation = table_annotation.fillna("undefined")

    table_annotation["augmentation"] = augmentation


    if augmentation == "wham":
        snr_sampler_wham = get_snr_sampler(distribution=args["snr_distribution"], snr_range=args["wham_snr_range"])
        
        noise_files = [
        os.path.join(root, f)
        for root, _, files in os.walk(wham)
        for f in files
        if f.endswith(".wav")
        ]
        # print(noise_files)
        table_annotation["noise"] = rd.choices(noise_files, k=len(table_annotation))
        table_annotation["snr_db"] = [snr_sampler_wham() for _ in range(len(table_annotation))]
    elif augmentation == "wind":
        snr_sampler_wind = get_snr_sampler(distribution=args["snr_distribution"], snr_range=args["wind_snr_range"])
        table_annotation["snr_db"] = [snr_sampler_wind() for _ in range(len(table_annotation))]
    elif augmentation == "saturation":
        gain_sampler_saturation = get_gain_sampler(distribution=args["snr_distribution"], gain_range=args["saturation_gain_range"])
        table_annotation["gain"] = [gain_sampler_saturation() for _ in range(len(table_annotation))]
    else:
        print(f"unknown augmentation {augmentation}")
    os.makedirs(f"{output_rootdir}/list", exist_ok=True)
    os.makedirs(f"{output_rootdir}/annotation", exist_ok=True)
    table_list["annotation_file"] = "annotation/annotation.csv"
    table_list.to_csv(f"{output_rootdir}/list/list.csv", sep=",", index=None)
    table_annotation.to_csv(f"{output_rootdir}/annotation/annotation.csv", sep=",", index=None)
    print(f"{output_rootdir}/annotation/annotation.csv")



