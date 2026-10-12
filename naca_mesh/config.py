"""Configuration for the current nested topology, preserving legacy defaults."""

import math
from dataclasses import dataclass
from .airfoil import coordinates, decode_airfoil
from .grading import (cosine_spacing, create_geometric_grading,
                      first_interval, progression_for_same_first_interval)

@dataclass
class Params:
    # Geometry
    W: float = 0.1
    H: float = 0.2
    h_in: float = 0.17
    Lx: float = 10.0
    Ly: float = 5.0

    # Gmsh curve node counts (legacy parameter convention)
    ny: int = 22
    nz_mid: int = 4
    nz_left_right: int = 12
    z_elements: int = 25    # uniform-Z layers for simple extrusion only
    upper_surface: int = 275
    lower_surface: int = 50
    nose_surface: int = 150

    # Wake params
    Ny_outer: int = 20
    Nx_far_wake: int = 50
    Nx_near_wake: int = 50
    near_wake_length: float = 0.02
    wake_stretch_in_Y: float = 1.25
    near_wake_offset_in_x_1: float = 1.0
    near_wake_offset_in_x_2: float = 1.0
    y_progression_near_wake: float = 1.2

    # Grading
    nose_bump_coeff: float = 3
    y_progression: float = 1.2
    y_outer_vertical_progression: float = 1.075
    y_outer_horizontal_progression: float = 1.0325

    # Airfoil params to create splines
    x_th: float = 0.1
    c: float = 1.0
    Nu: int = 500
    Nl: int = 500
    airfoil: str = "4412"
    closed_TE: bool = True

    # Existing fields specify Gmsh node counts; CLI surface inputs are elements.
    cosine_spacing = staticmethod(cosine_spacing)
    create_geometric_grading = staticmethod(create_geometric_grading)
    first_interval = staticmethod(first_interval)
    progression_for_same_first_interval = staticmethod(progression_for_same_first_interval)

    @property
    def m(self):
        return decode_airfoil(self.airfoil)[0]

    @property
    def p(self):
        return decode_airfoil(self.airfoil)[1]

    @property
    def t(self):
        return decode_airfoil(self.airfoil)[2]

    def coordinates(self, x):
        return coordinates(x, self.m, self.p, self.t, self.closed_TE)

    def __post_init__(self):
        self.validate()
        self.w_in, self.z1, self.z2, self.dz = self.uniform_z_segments()
        self.xU_cos = cosine_spacing(self.Nu + 1, 1.0, 0.0)
        self.xL_cos = cosine_spacing(self.Nl + 1, 0.0, 1.0)
        self.xu, self.yu, _, _ = self.coordinates(self.xU_cos)
        _, _, self.xl, self.yl = self.coordinates(self.xL_cos)
        self.outer_veritcal_ele, self.outer_vertical_heights = create_geometric_grading(
            self.Ny_outer, self.y_outer_vertical_progression)
        self.outer_horizontal_ele, self.outer_horizontal_heigths = create_geometric_grading(
            self.Nx_far_wake, self.y_outer_horizontal_progression,
            self.near_wake_length, self.Nx_near_wake)
        self.a_out_near_wake = first_interval(self.H, self.y_progression_near_wake, self.ny - 1)
        self.r_near_wake_in = progression_for_same_first_interval(
            self.h_in, self.ny - 1, self.a_out_near_wake)

    def validate(self):
        decode_airfoil(self.airfoil)
        if type(self.z_elements) is not int or self.z_elements < 1:
            raise ValueError("z_elements must be a positive integer")
        if self.c != 1.0:
            raise ValueError("nested topology currently requires c=1; chord scaling needs a separate refactor")
        if self.closed_TE is not True:
            raise ValueError("nested topology currently requires a closed trailing edge")
        for name in ("ny", "nz_mid", "nz_left_right", "upper_surface", "lower_surface",
                     "nose_surface", "Ny_outer", "Nx_far_wake", "Nx_near_wake", "Nu", "Nl"):
            value = getattr(self, name)
            if type(value) is not int or value < 2:
                raise ValueError(f"{name} must be an integer node/sampling count >= 2")
        for name in ("W", "H", "h_in", "Lx", "Ly", "wake_stretch_in_Y",
                     "nose_bump_coeff", "y_progression", "y_progression_near_wake",
                     "y_outer_vertical_progression", "y_outer_horizontal_progression"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not 0.05 < self.h_in < self.H < self.Ly:
            raise ValueError("nested topology requires 0.05 < h_in < H < Ly (existing nose offset)")
        if not 0 < self.x_th < 1 or not 0 < self.near_wake_length < 1:
            raise ValueError("x_th and near_wake_length must be within (0, 1)")
        for name in ("near_wake_offset_in_x_1", "near_wake_offset_in_x_2"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        wake_end = self.c + (self.Lx-self.c)*self.near_wake_length
        if self.Lx <= self.c or wake_end + self.near_wake_offset_in_x_1 + self.near_wake_offset_in_x_2 >= self.Lx:
            raise ValueError("Lx must leave room for the near wake and its offsets")


    def uniform_z_segments(self):
        """
        Compute uniform spacing across three horizontal segments:
        A–E (nz_left_right), E–F (nz_mid), F–B (nz_left_right).
        Returns (w_in, z1, z2, dz).
        """
        if self.nz_left_right < 2 or self.nz_mid < 2:
            raise ValueError("Need at least 2 nodes per curve.")
        denom = 2 * (self.nz_left_right - 1) + (self.nz_mid - 1)
        dz = self.W / denom
        z1 = dz * (self.nz_left_right - 1)
        w_in = dz * (self.nz_mid - 1)
        z2 = self.W - z1
        if not (0.0 < w_in < self.W):
            raise RuntimeError(f"Computed w_in={w_in} is invalid for W={self.W}.")
        if abs((2 * z1 + w_in) - self.W) > 1e-10:
            raise RuntimeError("Uniform-Z computation inconsistent.")
        return w_in, z1, z2, dz

    def grading(self, H_eff: float, h_eff: float):
        """
        Compute vertical grading parameters:
        - $$a_{\\text{out}}$$ = first interval on the outer verticals (using $$H_{\\text{eff}}$$ and $$r_{\\text{out}}$$),
        - $$r_{\\text{in}}$$ chosen so the inner verticals have the same first interval size $$a_{\\text{out}}$$ over $$h_{\\text{eff}}$$.
        Returns (r_out, a_out_near_wake, r_in).
        """
        if H_eff <= 0.0:
            raise ValueError("H_eff must be > 0.")
        if not (0.0 < h_eff < H_eff):
            raise ValueError("Require 0 < h_eff < H_eff.")
        r_out = self.y_progression
        a_out_near_wake = Params.first_interval(H_eff, r_out, self.ny - 1)
        r_in  = Params.progression_for_same_first_interval(h_eff, self.ny - 1, a_out_near_wake)
        return r_out, a_out_near_wake, r_in
