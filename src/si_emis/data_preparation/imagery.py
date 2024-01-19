"""
Shift satellite image (TIF format) by a certain offset in x and y direction.
This is required for some cases where the satellite image was obtained
some hours before or after the aircraft overpass and fast sea ice drift.

The shift offsets need to be specified for every satellite image inside the
constant dictionary in meters.

After the image is shifted, it gets cropped to the path coordinates.

User Notes
----------
Call the script again to crop the shifted images. Otherwise it uses the 
original images instead of the newly shifted ones.
"""

import argparse
import os
import shutil
import tempfile

import geopandas as gpd
import yaml
from dotenv import load_dotenv
from osgeo import gdal
from shapely.geometry import LineString
from shapely.ops import unary_union

load_dotenv()


SHIFT = {
    "aflux_rf08_case": {
        "img_orig": os.path.join(
            os.environ["PATH_SEC"],
            "data/sea_ice_emissivity/",
            "sentinel/AFLUX_RF08_20190331/rgb/rgb.tiff",
        ),
        "img_shift": os.path.join(
            os.environ["PATH_SEC"],
            "data/sea_ice_emissivity/",
            "sentinel/AFLUX_RF08_20190331/rgb_shifted.tiff",
        ),
        "img_crop": os.path.join(
            os.environ["PATH_SEC"],
            "data/sea_ice_emissivity/",
            "sentinel/AFLUX_RF08_20190331/rgb_shifted_crop.tiff",
        ),
        "dx": 0,
        "dy": 2500,
    },
    "aflux_rf08_lead": {
        "img_orig": os.path.join(
            os.environ["PATH_SEC"],
            "data/sea_ice_emissivity/",
            "sentinel/AFLUX_RF08_20190331/rgb/rgb.tiff",
        ),
        "img_shift": os.path.join(
            os.environ["PATH_SEC"],
            "data/sea_ice_emissivity/",
            "sentinel/AFLUX_RF08_20190331/rgb_shifted2.tiff",
        ),
        "img_crop": os.path.join(
            os.environ["PATH_SEC"],
            "data/sea_ice_emissivity/",
            "sentinel/AFLUX_RF08_20190331/rgb_shifted2_crop.tiff",
        ),
        "dx": 0,
        "dy": 4000,
    },
}


def shift():
    """
    Shifts all specified images
    """

    for case_name in SHIFT:
        # make sure that shifted file does not exist yet
        if os.path.exists(SHIFT[case_name]["img_shift"]):
            os.remove(SHIFT[case_name]["img_shift"])

        # copy original file
        shutil.copyfile(
            SHIFT[case_name]["img_orig"], SHIFT[case_name]["img_shift"]
        )

        # read copied original file
        ds_orig = gdal.Open(SHIFT[case_name]["img_shift"])

        # get transform of input file and apply shift
        gt = ds_orig.GetGeoTransform()

        print(
            """
            top-left x: {}
            w-e pixel size: {}
            rotation:{}
            top-left y: {}
            rotation: {}
            n-s pixel size: {}
            """.format(
                *gt
            )
        )

        gtl = list(gt)
        gtl[0] += SHIFT[case_name]["dx"]
        gtl[3] += SHIFT[case_name]["dy"]
        gtl = tuple(gtl)

        # write to new file and set new transform
        ds_shifted = gdal.Open(SHIFT[case_name]["img_shift"])
        ds_shifted.SetGeoTransform(gtl)


def crop():
    """
    Crop satellite imagery to rectangular area between two coordinates. The
    coordinates could be the start and end point of a flight segment. The
    width of the crop needs to be specified. The length of the crop is given
    by the distance of the two coordinates.
    """

    for case_name in SHIFT:
        # skip cases without crop output
        if "img_crop" not in SHIFT[case_name]:
            continue

        # read path coordinates
        with open("./src/si_emis/data_preparation/coordinates.yaml") as f:
            coordinates = yaml.safe_load(f)[case_name]

        with tempfile.TemporaryDirectory(dir=os.environ["PATH_CACHE"]) as td:
            # buffer flight track over land and export as shapefile
            gdf_gps = gpd.GeoDataFrame(
                crs="epsg:4326",
                geometry=[LineString([coordinates["p1"], coordinates["p2"]])],
            ).to_crs(epsg="32631")
            gdf_gps["geometry"] = gdf_gps.geometry.buffer(
                distance=coordinates["crop_width"], cap_style=2
            )
            gdf_gps = gpd.GeoSeries(unary_union(gdf_gps["geometry"]))

            file_track = os.path.join(td, "track.shp")
            gdf_gps.to_file(file_track, crs="EPSG:32631")

            # extract raster along track
            gdal.Warp(
                SHIFT[case_name]["img_crop"],
                SHIFT[case_name]["img_shift"],
                cutlineDSName=file_track,
                cropToCutline=True,
                format="GTiff",
            )


if __name__ == "__main__":
    # select via command line which function to run
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, required=True)
    args = parser.parse_args()

    if args.task == "shift":
        shift()
    else:
        crop()
