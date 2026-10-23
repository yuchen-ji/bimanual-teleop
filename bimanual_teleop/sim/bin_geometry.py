"""Dimensions of the supplied open crate after fitting its measured interior.

The source CAD has a 449 x 322 mm clear opening, an 8 mm internal floor,
and a 182 mm rim. Sizes below describe that mesh, not a solid collision box.
"""

from ._dependencies import np


CAD_OPENING_M = (.449, .322)
CAD_FLOOR_HEIGHT_M = .008
CAD_RIM_HEIGHT_M = .182
CAD_OUTER_SIZE_M = (.493, .348, .182)
CAD_OUTER_RIM_SIZE_M = (.475, .348, .182)


def bin_geometry(scene):
    """Return the shared visual/collision scale and actual floor/rim offsets.

    ``bin_centers`` locate the bottom of the top crate, while
    ``bin_inner_size`` is clear opening length, width, and floor-to-rim depth.
    """
    source_inner = np.array([*CAD_OPENING_M, CAD_RIM_HEIGHT_M-CAD_FLOOR_HEIGHT_M])
    scale = np.asarray(scene["bin_inner_size"], dtype=float)/source_inner
    return {"scale": scale, "outer_size_m": np.asarray(CAD_OUTER_SIZE_M)*scale,
            "outer_rim_size_m": np.asarray(CAD_OUTER_RIM_SIZE_M)*scale,
            "floor_height_m": float(CAD_FLOOR_HEIGHT_M*scale[2]),
            "rim_height_m": float(CAD_RIM_HEIGHT_M*scale[2])}
