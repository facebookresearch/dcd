#  Copyright (c) 2015 Peter Onrejka
#
# Licensed under the GNU General Public License;
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://opensource.org/licenses/gpl-license
#
# This file is a modified version of
# https://github.com/sical/polygons_complexity/blob/master/complexity.py
# which itself is based on
# https://github.com/pondrejk/PolygonComplexity/blob/master/PolygonComplexity.py


import math
import os
import pandas as pd
import geopandas as gpd
import shapely


def get_notches(poly):
    """
    Determine the number of notches in a polygon object and calculate
    normalized notches of polygon

    Based on:
        "Measuring the Complexity of Polygonal Objects"
        (Thomas Brinkhoff, Hans-Peter Kriegel, Ralf Schneider, Alexander Braun)
        http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.73.1045&rep=rep1&type=pdf

        https://github.com/pondrejk/PolygonComplexity/blob/master/PolygonComplexity.py

    @poly (Shapely Polygon object)

    Returns normalized notches
    """
    notches = 0
    coords = list(poly.exterior.coords)
    for i, pt in enumerate(coords[:-1]):
        x_diff = coords[i + 1][0] - pt[0]
        y_diff = coords[i + 1][1] - pt[1]
        angle = math.atan2(y_diff, x_diff)
        if angle < 0:
            angle += 2 * math.pi
        if angle > math.pi:
            notches += 1

    if notches != 0:
        notches_norm = notches / (len(coords) - 3)
    else:
        notches_norm = 0

    return notches_norm


def get_stats(gdf, coeff_ampl, coeff_conv):
    """
    Get polygon's amplitude of vibration:

    ampl(pol) = (boundary(pol) - boundary(convexhull(pol))) / boundary(pol)

    Get deviation from convex hull:
    conv(pol) = (area(convexhull(pol)) - area(pol)) / area(convexhull(pol))

    Measure complexity

     Based on:
        "Measuring the Complexity of Polygonal Objects"
        (Thomas Brinkhoff, Hans-Peter Kriegel, Ralf Schneider, Alexander Braun)
        http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.73.1045&rep=rep1&type=pdf

        https://github.com/pondrejk/PolygonComplexity/blob/master/PolygonComplexity.py

    Get area, centroid, distance from each others, boudary, convex hull,
    perimeter, number of vertices.

    Returns tuple with dict of stats values and GeoDataframe with stats
    """
    nb = gdf["geometry"].count()
    gdf["area"] = gdf["geometry"].area
    tot_area = gdf["area"].sum()
    gdf["centroid"] = gdf["geometry"].centroid
    gdf["boundary"] = gdf["geometry"].boundary
    gdf["convex_hull"] = gdf["geometry"].convex_hull
    gdf["convex_boundary"] = gdf["geometry"].convex_hull.boundary
    gdf["convex_area"] = gdf["geometry"].convex_hull.area
    gdf["nbvertices"] = gdf["geometry"].apply(lambda x: len(list(x.exterior.coords)))
    gdf["notches"] = gdf["geometry"].apply(lambda x: get_notches(x))

    gdf["amplitude"] = gdf.apply(
        lambda x: (x["boundary"].length - x["convex_boundary"].length)
        / (x["boundary"].length + 1e-3),
        axis=1,
    )
    gdf["convex"] = gdf.apply(
        lambda x: (x["convex_area"] - x["area"]) / (x["convex_area"] + 1e-3), axis=1
    )
    gdf["complexity"] = gdf.apply(
        lambda x: coeff_ampl * x["amplitude"] * x["notches"] + coeff_conv * x["convex"],
        axis=1,
    )

    mean_amplitude = gdf["amplitude"].mean()
    mean_convex = gdf["convex"].mean()
    mean_norm_notches = gdf["notches"].mean()
    mean_complexity = gdf["complexity"].mean()

    gdf["perimeter"] = gdf["geometry"].length
    tot_perimeter = gdf["perimeter"].sum()

    if ("lat" in gdf.columns) or ("lon" in gdf.columns):
        columns_drop = [
            "boundary",
            "convex_hull",
            "convex_boundary",
            "convex_area",
            "centroid",
            "lat",
            "lon",
        ]
    else:
        columns_drop = [
            "boundary",
            "convex_hull",
            "convex_boundary",
            "convex_area",
            "centroid",
        ]
    gdf = gdf.drop(columns_drop, axis=1)

    gdf = gdf.reset_index()

    if nb > 1:
        gdf = gdf.sort_values(by="perimeter", ascending=False)
        gdf = gdf.iloc[[0]]

    return {
        "area": tot_area,
        "perimeter": tot_perimeter,
        "amplitude": mean_amplitude,
        "convex": mean_convex,
        "notches": mean_norm_notches,
        "complexity": mean_complexity,
    }, gdf


def complexity(points, coeff_ampl=0.8, coeff_conv=0.2):
    polygon = shapely.geometry.Polygon(points)
    gdf = gpd.GeoDataFrame(geometry=gpd.GeoSeries([polygon]))
    dict_complexity, gdf = get_stats(gdf, coeff_ampl, coeff_conv)

    return dict_complexity
