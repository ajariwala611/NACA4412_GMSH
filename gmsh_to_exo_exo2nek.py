#!/usr/bin/env python3
"""
Convert a Gmsh .msh mesh into an Exodus .exo file for exo2nek.

Key behavior:
- Writes only volume elements to Exodus (no boundary tri/quad element blocks).
- Reconstructs Exodus side sets from Gmsh physical SURFACE groups.
- Renumbers side set IDs to 1..N (continuous), as required by NekRS/Nek5000 tools.
"""

import argparse
import os
from collections import OrderedDict

import meshio
import netCDF4
import numpy as np


# Exodus side numbering in terms of corner-node indices for each volume element.
# Node order follows the meshio canonical order for each cell type.
FACE_CORNERS = {
    "tetra": [
        [0, 1, 3],  # side 1
        [1, 2, 3],  # side 2
        [0, 3, 2],  # side 3
        [0, 2, 1],  # side 4
    ],
    "tetra10": [
        [0, 1, 3],
        [1, 2, 3],
        [0, 3, 2],
        [0, 2, 1],
    ],
    "wedge": [
        [0, 1, 4, 3],  # side 1
        [1, 2, 5, 4],  # side 2
        [0, 3, 5, 2],  # side 3
        [0, 2, 1],     # side 4
        [3, 4, 5],     # side 5
    ],
    "wedge15": [
        [0, 1, 4, 3],
        [1, 2, 5, 4],
        [0, 3, 5, 2],
        [0, 2, 1],
        [3, 4, 5],
    ],
    "hexahedron": [
        [0, 1, 5, 4],  # side 1
        [1, 2, 6, 5],  # side 2
        [2, 3, 7, 6],  # side 3
        [0, 4, 7, 3],  # side 4
        [0, 3, 2, 1],  # side 5
        [4, 5, 6, 7],  # side 6
    ],
    "hexahedron20": [
        [0, 1, 5, 4],
        [1, 2, 6, 5],
        [2, 3, 7, 6],
        [0, 4, 7, 3],
        [0, 3, 2, 1],
        [4, 5, 6, 7],
    ],
    "hexahedron27": [
        [0, 1, 5, 4],
        [1, 2, 6, 5],
        [2, 3, 7, 6],
        [0, 4, 7, 3],
        [0, 3, 2, 1],
        [4, 5, 6, 7],
    ],
}

VOLUME_TYPES = set(FACE_CORNERS.keys())
SURFACE_TYPES = {"triangle", "triangle6", "quad", "quad8", "quad9"}

CORNER_COUNT = {
    "triangle": 3,
    "triangle6": 3,
    "quad": 4,
    "quad8": 4,
    "quad9": 4,
    "tetra": 4,
    "tetra10": 4,
    "wedge": 6,
    "wedge15": 6,
    "hexahedron": 8,
    "hexahedron20": 8,
    "hexahedron27": 8,
}


def _face_key(nodes, n_corners):
    # Use corner nodes only and sort to make orientation-independent matching.
    return tuple(sorted(int(n) for n in nodes[:n_corners]))


def _build_volume_face_map(volume_blocks):
    """
    Build:
    - face_map: face-key -> list[(global_elem_id, local_side_id)]
    - global element numbering follows Exodus block order.
    """
    face_map = {}
    global_elem = 1
    for cell_type, connectivity in volume_blocks:
        faces = FACE_CORNERS[cell_type]
        for elem_nodes in connectivity:
            for local_side, corner_ids in enumerate(faces, start=1):
                key = tuple(sorted(int(elem_nodes[i]) for i in corner_ids))
                face_map.setdefault(key, []).append((global_elem, local_side))
            global_elem += 1
    return face_map


def _ordered_surface_phys_groups(field_data, surface_dim=2):
    """
    Returns OrderedDict: phys_id -> name, ordered by phys_id.
    """
    items = []
    for name, value in field_data.items():
        # meshio field_data uses [id, dim]
        phys_id = int(value[0])
        dim = int(value[1])
        if dim == surface_dim:
            items.append((phys_id, name))
    items.sort(key=lambda x: x[0])
    return OrderedDict(items)


def _split_volume_mesh(mesh):
    """
    Keep only volume cell blocks and aligned cell data.
    """
    keep = []
    for i, block in enumerate(mesh.cells):
        if block.type in VOLUME_TYPES:
            keep.append(i)

    if not keep:
        raise RuntimeError("No supported volume elements found (tet/wedge/hex).")

    volume_cells = [mesh.cells[i] for i in keep]

    volume_cell_data = {}
    for key, data_list in mesh.cell_data.items():
        volume_cell_data[key] = [data_list[i] for i in keep]

    return volume_cells, volume_cell_data


def _build_sidesets(mesh, volume_blocks, volume_block_indices):
    """
    Build Exodus side sets from Gmsh physical surface groups.
    Returns list of dict:
      {name, old_id, new_id, elem_ids(np.int32), side_ids(np.int32), unmatched, ambiguous}
    """
    if "gmsh:physical" not in mesh.cell_data:
        raise RuntimeError("Input mesh has no gmsh:physical cell data.")

    surface_groups = _ordered_surface_phys_groups(mesh.field_data, surface_dim=2)
    if not surface_groups:
        raise RuntimeError("No physical surface groups found in field_data.")

    face_map = _build_volume_face_map(volume_blocks)
    phys_data = mesh.cell_data["gmsh:physical"]

    sidesets = []
    new_id = 1

    # Iterate groups in ascending original physical ID order.
    for old_id, name in surface_groups.items():
        elem_ids = []
        side_ids = []
        unmatched = 0
        ambiguous = 0

        for block_i, block in enumerate(mesh.cells):
            if block_i in volume_block_indices:
                continue
            if block.type not in SURFACE_TYPES:
                continue

            block_phys = phys_data[block_i]
            n_corner = CORNER_COUNT[block.type]
            matched_rows = np.where(block_phys == old_id)[0]

            for row in matched_rows:
                key = _face_key(block.data[row], n_corner)
                owners = face_map.get(key, [])
                if not owners:
                    unmatched += 1
                    continue
                if len(owners) > 1:
                    # Should be rare for boundary groups; still pick first deterministically.
                    ambiguous += 1
                owner = owners[0]
                elem_ids.append(owner[0])
                side_ids.append(owner[1])

        if elem_ids:
            # Remove duplicates while preserving order.
            seen = set()
            dedup_elem = []
            dedup_side = []
            for e, s in zip(elem_ids, side_ids):
                key = (int(e), int(s))
                if key in seen:
                    continue
                seen.add(key)
                dedup_elem.append(e)
                dedup_side.append(s)

            sidesets.append(
                {
                    "name": name,
                    "old_id": old_id,
                    "new_id": new_id,
                    "elem_ids": np.asarray(dedup_elem, dtype=np.int32),
                    "side_ids": np.asarray(dedup_side, dtype=np.int32),
                    "unmatched": unmatched,
                    "ambiguous": ambiguous,
                }
            )
            new_id += 1

    if not sidesets:
        raise RuntimeError(
            "No side sets could be constructed from physical surface groups."
        )

    return sidesets


def _write_sidesets_to_exodus(exo_file, sidesets):
    with netCDF4.Dataset(exo_file, "a") as ds:
        nss = len(sidesets)

        if "num_side_sets" in ds.dimensions:
            raise RuntimeError(
                "Output Exodus file already has num_side_sets. "
                "Please choose a new output filename."
            )

        # Name width compatibility with existing meshio exodus files.
        if "len_string" in ds.dimensions:
            name_width = len(ds.dimensions["len_string"])
            name_dim = "len_string"
        else:
            name_width = 33
            name_dim = "len_name"
            ds.createDimension(name_dim, name_width)

        ds.createDimension("num_side_sets", nss)
        ss_status = ds.createVariable("ss_status", "i4", ("num_side_sets",))
        ss_prop1 = ds.createVariable("ss_prop1", "i4", ("num_side_sets",))
        ss_prop1.setncattr("name", "ID")
        ss_names = ds.createVariable("ss_names", "S1", ("num_side_sets", name_dim))
        ss_names.set_auto_mask(False)

        for i, ss in enumerate(sidesets, start=1):
            idx = i - 1
            ss_status[idx] = 1
            ss_prop1[idx] = int(ss["new_id"])

            name = ss["name"][: max(0, name_width - 1)]
            name_bytes = name.encode("ascii", errors="ignore")
            ss_names[idx, :] = b"\x00"
            for j, b in enumerate(name_bytes):
                ss_names[idx, j] = bytes([b])

            dim_side = f"num_side_ss{i}"
            ds.createDimension(dim_side, len(ss["elem_ids"]))

            elem_var = ds.createVariable(f"elem_ss{i}", "i4", (dim_side,))
            side_var = ds.createVariable(f"side_ss{i}", "i4", (dim_side,))
            elem_var[:] = ss["elem_ids"]
            side_var[:] = ss["side_ids"]


def _rewrite_exodus_as_netcdf3(src_path, dst_path):
    """
    Re-encode Exodus as NetCDF-3 64-bit offset.

    This is needed for older exo2nek builds that cannot open NetCDF-4/HDF5 Exodus files.
    """
    with netCDF4.Dataset(src_path, "r") as src, netCDF4.Dataset(
        dst_path, "w", format="NETCDF3_64BIT_OFFSET"
    ) as dst:
        # Global attributes
        for attr in src.ncattrs():
            dst.setncattr(attr, src.getncattr(attr))

        # Dimensions: keep only one unlimited dimension (time_step).
        dropped_dims = set()
        for name, dim in src.dimensions.items():
            if dim.isunlimited():
                if name == "time_step":
                    dst.createDimension(name, None)
                elif len(dim) == 0:
                    # meshio may create zero-length unlimited num_node_sets in
                    # NetCDF-4. NetCDF-3 allows only one unlimited dimension.
                    dropped_dims.add(name)
                else:
                    dst.createDimension(name, len(dim))
            else:
                dst.createDimension(name, len(dim))

        # Variables
        for name, var in src.variables.items():
            if any(dim_name in dropped_dims for dim_name in var.dimensions):
                continue

            fill_value = var.getncattr("_FillValue") if "_FillValue" in var.ncattrs() else None
            if fill_value is None:
                out_var = dst.createVariable(name, var.datatype, var.dimensions)
            else:
                out_var = dst.createVariable(
                    name, var.datatype, var.dimensions, fill_value=fill_value
                )

            for attr in var.ncattrs():
                if attr == "_FillValue":
                    continue
                out_var.setncattr(attr, var.getncattr(attr))

            out_var[:] = var[:]


def convert(infile, outfile, verbose=True, netcdf3=True):
    mesh = meshio.read(infile)

    volume_cells, volume_cell_data = _split_volume_mesh(mesh)
    volume_block_indices = [
        i for i, block in enumerate(mesh.cells) if block.type in VOLUME_TYPES
    ]
    volume_blocks = [(b.type, b.data) for b in volume_cells]

    sidesets = _build_sidesets(mesh, volume_blocks, volume_block_indices)

    # Write a volume-only exodus mesh first (NetCDF-4 temp file).
    out_mesh = meshio.Mesh(
        points=mesh.points,
        cells=volume_cells,
        cell_data=volume_cell_data,
    )
    tmp_out = outfile + ".tmpnc4"
    meshio.write(tmp_out, out_mesh, file_format="exodus")

    # Append side sets.
    _write_sidesets_to_exodus(tmp_out, sidesets)

    if netcdf3:
        _rewrite_exodus_as_netcdf3(tmp_out, outfile)
        os.remove(tmp_out)
    else:
        os.replace(tmp_out, outfile)

    if verbose:
        print(f"Input:  {infile}")
        print(f"Output: {outfile}")
        print(f"Volume blocks written: {len(volume_cells)}")
        print("Side set mapping (Gmsh physical -> Exodus side set ID):")
        for ss in sidesets:
            print(
                f"  old_id={ss['old_id']:>3} name={ss['name']:<24} "
                f"-> new_id={ss['new_id']:>2} sides={len(ss['elem_ids'])}"
            )
            if ss["unmatched"] > 0 or ss["ambiguous"] > 0:
                print(
                    f"    note: unmatched={ss['unmatched']}, ambiguous={ss['ambiguous']}"
                )


def main():
    parser = argparse.ArgumentParser(
        description="Convert Gmsh .msh to Exodus .exo with side sets for exo2nek."
    )
    parser.add_argument("infile", help="Input Gmsh mesh file (.msh)")
    parser.add_argument("outfile", help="Output Exodus file (.exo or .e)")
    parser.add_argument(
        "--netcdf4",
        action="store_true",
        help="Keep NetCDF-4/HDF5 Exodus output (default is NetCDF-3 64-bit offset).",
    )
    parser.add_argument(
        "--quiet", action="store_true", help="Suppress summary output."
    )
    args = parser.parse_args()

    convert(
        args.infile,
        args.outfile,
        verbose=not args.quiet,
        netcdf3=not args.netcdf4,
    )


if __name__ == "__main__":
    main()
