import argparse
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import os.path as osp
import pickle
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import scipy.ndimage.interpolation
import SimpleITK as sitk
import tigre
import yaml
from tigre.utilities import CTnoise, gpu
from tigre.utilities.geometry import Geometry


def config_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--configPath",
        default="./dataGenerator/config_2views.yml",
        type=str,
        help="Path of config file",
    )
    parser.add_argument(
        "--dataPath",
        default="/home/public/CTSpine1K/data/data-MHD_ctpro_woMask1",
        type=str,
        help="Path of output data",
    )
    parser.add_argument(
        "--case",
        default="volume-covid19-A-0377_ct",
        type=str,
        help="Path of case folder, which should contain ct_file.mha",
    )

    parser.add_argument(
        "--outputName", default="data_2views", type=str, help="Name of output data"
    )
    return parser


def main():
    parser = config_parser()
    args = parser.parse_args()
    configPath = Path(args.configPath)
    case = Path(args.dataPath) / Path(args.case)
    matPath = str(case / "ct_file.mha")
    outputPath = case / f"{args.outputName}.pickle"
    generator(matPath, configPath, outputPath, True)


# %% Geometry
class ConeGeometry_special(Geometry):
    """
    Cone beam CT geometry.
    """

    def __init__(self, data):
        Geometry.__init__(self)

        # VARIABLE                                          DESCRIPTION                    UNITS
        # -------------------------------------------------------------------------------------
        self.DSD = data["DSD"] / 1000  # Distance Source Detector      (m)
        self.DSO = data["DSO"] / 1000  # Distance Source Origin        (m)
        # Detector parameters
        self.nDetector = np.array(
            data["nDetector"]
        )  # number of pixels              (px)
        self.dDetector = (
            np.array(data["dDetector"]) / 1000
        )  # size of each pixel            (m)
        self.sDetector = (
            self.nDetector * self.dDetector
        )  # total size of the detector    (m)
        # Image parameters
        self.nVoxel = np.array(
            data["nVoxel"][::-1]
        )  # number of voxels              (vx)
        self.dVoxel = (
            np.array(data["dVoxel"][::-1]) / 1000
        )  # size of each voxel            (m)
        self.sVoxel = self.nVoxel * self.dVoxel  # total size of the image       (m)

        # Offsets
        self.offOrigin = (
            np.array(data["offOrigin"][::-1]) / 1000
        )  # Offset of image from origin   (m)
        self.offDetector = (
            np.array([data["offDetector"][1], data["offDetector"][0], 0]) / 1000
        )  # Offset of Detector            (m)

        # Auxiliary
        self.accuracy = data[
            "accuracy"
        ]  # Accuracy of FWD proj          (vx/sample)  # noqa: E501
        # Mode
        self.mode = data["mode"]  # parallel, cone                ...
        self.filter = data["filter"]


def convert_to_attenuation(
    data: np.array, rescale_slope: float, rescale_intercept: float
):
    """
    CT scan is measured using Hounsfield units (HU). We need to convert it to attenuation.

    The HU is first computed with rescaling parameters:
        HU = slope * data + intercept

    Then HU is converted to attenuation:
        mu = mu_water + HU/1000x(mu_water-mu_air)
        mu_water = 0.206
        mu_air=0.0004

    Args:
    data (np.array(X, Y, Z)): CT data.
    rescale_slope (float): rescale slope.
    rescale_intercept (float): rescale intercept.

    Returns:
    mu (np.array(X, Y, Z)): attenuation map.

    """
    HU = data * rescale_slope + rescale_intercept
    mu_water = 0.206
    mu_air = 0.0004
    mu = mu_water + (mu_water - mu_air) / 1000 * HU
    # mu = mu * 100
    return mu


def loadImage(
    test_data,
    zoom_factor,
    nVoxels,
    convert,
    rescale_slope,
    rescale_intercept,
    normalize=True,
    percentile=False,
    min=None,
    max=None,
):
    """
    Load CT image.
    """

    # Loads data in F_CONTIGUOUS MODE (column major), convert to Row major
    image_ori = test_data
    if convert:
        print("Convert from HU to attenuation")
        image = convert_to_attenuation(image_ori, rescale_slope, rescale_intercept)
    else:
        image = image_ori

    imageDim = image.shape

    if zoom_factor != 1.0:
        print(
            f"Resize ct image from {imageDim[0]}x{imageDim[1]}x{imageDim[2]} to "
            f"{nVoxels[0]}x{nVoxels[1]}x{nVoxels[2]}"
        )
        image = scipy.ndimage.zoom(
            image, (zoom_factor, zoom_factor, zoom_factor), order=3, prefilter=False
        )

    image_max = np.max(image)
    image_min = np.min(image)
    image_mean = np.mean(image)
    print(
        "Range of CT image is [%f, %f], mean: %f" % (image_min, image_max, image_mean)
    )
    if normalize and image_min != 0 and image_max != 1:
        print("Normalize range to [0, 1]")
        if percentile:
            p1 = np.percentile(image, 0.05)
            p99 = np.percentile(image, 99.9)
            image = np.clip(image, p1, None)
            image_min = p1
            image_max = p99
        else:
            image_min = min if min is not None else image_min
            image_max = max if max is not None else image_max
            image = np.clip(image, image_min, None)
        image = (image - image_min) / (image_max - image_min)

    return image


def calc_nDetector(DSD, DSO, nVoxel, dVoxel, dDetector):
    """
    Calculate number of detector pixels using similar similar triangles:
    """
    # nDetector_W/nVoxel_W = DSD/(DSO-nVoxel_W*dVoxel_W/2)
    nVoxel_W = nVoxel[0]
    nDetector_W = np.round(nVoxel_W * DSD / (DSO - nVoxel_W * dVoxel[0] / 2)).astype(
        int
    )
    # nDetector_H/nVoxel_H = DSD/(DSO-nVoxel_H*dVoxel/2)
    nVoxel_H = nVoxel[-1]
    nDetector_H = np.round(nVoxel_H * DSD / (DSO - nVoxel_H * dVoxel[-1] / 2)).astype(
        int
    )
    return [nDetector_H, nDetector_W]


def generator(matPath, configPath, outputPath, show=True):
    """
    Generate projections given CT image and configuration.

    """

    # Load configuration
    with open(configPath, "r") as handle:
        data = yaml.safe_load(handle)
    # load ct image
    test_data = sitk.GetArrayFromImage(sitk.ReadImage(matPath))  # z,y,x
    test_data = np.transpose(test_data, (2, 1, 0)).astype(np.float32)  # x,y,z
    # overwrite nVoxel
    target_img_size = 256
    zoom_factor = target_img_size / test_data.shape[0]
    data["nVoxel"] = [
        target_img_size,
        target_img_size,
        round(test_data.shape[2] * zoom_factor),
    ]

    # proprocessing
    img = loadImage(
        test_data,
        zoom_factor,
        data["nVoxel"],
        data["convert"],
        data["rescale_slope"],
        data["rescale_intercept"],
        data["normalize"],
        data["percentile"],
        data["min"],
        data["max"],
    )
    data["image"] = img.copy()
    # calc nDetector
    data["nDetector"] = calc_nDetector(
        data["DSD"], data["DSO"], data["nVoxel"], data["dVoxel"], data["dDetector"]
    )

    # plt.figure()
    # plt.imshow(img[:,:,0])
    # plt.show()
    geo = ConeGeometry_special(data)
    # Generate training images
    if data["randomAngle"] is False:
        data["train"] = {
            "angles": np.linspace(
                0, data["totalAngle"] / 180 * np.pi, data["numTrain"] + 1
            )[:-1]
            + data["startAngle"] / 180 * np.pi
        }
    else:
        data["train"] = {
            "angles": np.sort(
                np.random.rand(data["numTrain"]) * data["totalAngle"] / 180 * np.pi
            )
            + data["startAngle"] / 180 * np.pi
        }
    gpuids = gpu.GpuIds("NVIDIA GeForce RTX 3090")
    print("Use GPUs: ", gpuids)
    projections = tigre.Ax(
        np.transpose(img, (2, 1, 0)).copy(),
        geo,
        data["train"]["angles"],
        gpuids=gpuids,
    )[:, ::-1, :]
    if data["noise"] and data["normalize"]:
        print("Add noise to projections")
        noise_projections = CTnoise.add(
            projections, Poisson=1e5, Gaussian=np.array([0, 10])
        )
        noise_projections[noise_projections < 0.0] = 0.0
        data["train"]["projections"] = noise_projections
    else:
        data["train"]["projections"] = projections

    # Generate validation images
    if "2view" in configPath.name:
        data["val"] = {
            "angles": np.linspace(
                0, data["totalAngle"] / 180 * np.pi, data["numTrain"] + 1
            )[:-1]
            + data["startAngle"] / 180 * np.pi
        }
    else:
        data["val"] = {
            "angles": np.sort(np.random.rand(data["numVal"]) * 180 / 180 * np.pi)
            + data["startAngle"] / 180 * np.pi
        }
    projections = tigre.Ax(
        np.transpose(img, (2, 1, 0)).copy(),
        geo,
        data["val"]["angles"],
        gpuids=gpuids,
    )[:, ::-1, :]
    if data["noise"] != 0 and data["normalize"]:
        print("Add noise to projections")
        noise_projections = CTnoise.add(
            projections, Poisson=1e5, Gaussian=np.array([0, data["noise"]])
        )
        data["val"]["projections"] = noise_projections
    else:
        data["val"]["projections"] = projections

    if show:
        # print("Display ct image")
        # tigre.plotimg(img.transpose((2, 0, 1)), dim="z")
        # print("Display training images")
        # tigre.plotproj(data["train"]["projections"][:, ::-1, :])
        # print("Display validation images")
        # tigre.plotproj(data["val"]["projections"][:, ::-1, :])
        # save to tmp/
        os.makedirs("tmp", exist_ok=True)
        tigre.plotimg(
            img.transpose((2, 0, 1)), dim="z", savegif=outputPath.parent / f"image.gif"
        )
        tigre.plotproj(
            data["train"]["projections"][:, ::-1, :],
            savegif=outputPath.parent / f"train_projections.gif",
        )
        tigre.plotproj(
            data["val"]["projections"][:, ::-1, :],
            savegif=outputPath.parent / f"val_projections.gif",
        )

    # Save data
    os.makedirs(osp.dirname(outputPath), exist_ok=True)
    with open(outputPath, "wb") as handle:
        pickle.dump(data, handle, pickle.HIGHEST_PROTOCOL)

    print(f"Save files in {outputPath}")


if __name__ == "__main__":
    main()
