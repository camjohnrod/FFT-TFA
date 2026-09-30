# The Fourier transform over the partition lattice (ER-14a). The kernel and every partition field are real, so
# only the non-redundant half of the frequencies is stored and used.

import numpy as np

import config

def get_P0_transformed(P0_offset_blocks):
    # P̂0(ξ) in ER-14a, transformed over the two in-plane lattice directions. The kernel is real, so P̂0(-ξ) is the
    # complex conjugate of P̂0(ξ), and only the non-redundant half of the frequencies is kept (rfftn).
    P0_offset_lattice = P0_offset_blocks.reshape(config.partition_number_per_side, config.partition_number_per_side,
                                                 6, 6)
    return np.fft.rfftn(P0_offset_lattice, axes=(0, 1))

def apply_on_partition_lattice(fourier_blocks, partition_field):
    # Applies a translation-invariant operator, given by its 6 × 6 block at every lattice frequency, to a
    # partition-stacked field: transform the field, multiply frequency by frequency, transform back (ER-14a).
    # The field and the operator are real, so only the non-redundant half of the frequencies is used.
    lattice_shape = (config.partition_number_per_side, config.partition_number_per_side)
    field_transformed = np.fft.rfftn(partition_field.reshape(*lattice_shape, 6), axes=(0, 1))
    result_transformed = np.einsum('...ij,...j->...i', fourier_blocks, field_transformed)
    return np.fft.irfftn(result_transformed, s=lattice_shape, axes=(0, 1)).reshape(config.partition_count, 6)
