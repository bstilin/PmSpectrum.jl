# Copyright (c) 2026 Ben Stilin
#
# Licensed under the MIT License. See LICENSE for details.

module PmSpectrum

using DoubleFloats


module Utils
   using ..DoubleFloats
   using LinearAlgebra
   using GenericLinearAlgebra  # extends svdvals for Double64 / BigFloat matrices
   using GenericSchur          # extends eigen for Double64 / BigFloat matrices

   include("Utils.jl")
   include("BranchInverse.jl")
   include("ConstructPerturbationSolution.jl")
   include("IntervalHelpers.jl")
end

module Bases
   using Base.Threads
   using ..DoubleFloats
   using ..Utils

   include("BSplineBasis.jl")
   include("BSplineGalerkinMatrix.jl")
   include("DensityIntegration.jl")
   include("Correlation.jl")
   include("BlockMatrix.jl")

   # Not exported: BlockMatrix.jl is unvetted. Use Bases.project_global_approximation.
   export SingleBSpline, convert_precision, merge_breakpoints
   export build_integration_mesh, integrate_on_mesh, DensityQuadrature, density_quadrature,
          integrate_against_density, total_mass
   export CorrelationCurve, correlation_curve, estimate_decay_rate
end


module SchrodingerFD
   using LinearAlgebra: SymTridiagonal, eigen

   include("SchrodingerFD.jl")

   # Not exported: `potential` is too generic a name. Use SchrodingerFD.potential.
   export solve_fd, assemble_fd_operator, fd_laplacian
end

module SchrodingerShooting
   using ..SchrodingerFD: potential

   using OrdinaryDiffEqRKN: DPRKN12, SecondOrderODEProblem, solve
   using Roots: find_zero, A42, Bisection, Tracks
   using QuadGK: quadgk

   include("SchrodingerShooting.jl")

   # Not exported: left_initial_data, right_initial_data, phase_state are helpers, not API.
   export shooting_data, matching_residual, refine_eigenvalue,
          shooting_eigenfunction, turning_point, bracket_from_fd,
          brackets_from_fd, refine_spectrum, merge_parity
end

module TransferPrediction
   using ..Utils: boundary_layer_scale

   include("TransferPrediction.jl")

   export psi_weight, psi_cutoff, extend_parity,
          layer_scale, sd_from_halfwidth, halfwidth_from_sd,
          predicted_eigenvalue, predicted_eigenvalues,
          recovered_eigenvalue, recovered_eigenvalues,
          predicted_eigenfunction, predict_transfer_mode, predict_transfer_spectrum
end

# Bring Bases' exported names into scope so the top-level re-exports below resolve.
# The solver and prediction modules are deliberately not brought in: they are reached
# by name, as SchrodingerFD.solve_fd, TransferPrediction.predicted_eigenvalues, and so on.
using .Bases

include("Results.jl")
include("BatchExperiments.jl")

# Standalone diagnostic; nothing else in the package depends on it.
include("ResolventNorm.jl")

# --- RE-EXPORT FOR THE USER ---

export Double64
export Utils, Bases, SchrodingerFD, SchrodingerShooting, TransferPrediction
export ResolventNorm
export InvariantDensityResult, RehydratedResult, save_result, load_result, rehydrate
export ExperimentEntry, run_single_experiment, run_grid
export load_manifest, filter_results, load_results, show_manifest
export SingleBSpline, convert_precision, merge_breakpoints
export build_integration_mesh, integrate_on_mesh, DensityQuadrature, density_quadrature,
       integrate_against_density, total_mass
export CorrelationCurve, correlation_curve, estimate_decay_rate

# The Schrodinger solvers and the transfer-prediction routines are not re-exported.
# They are specialized enough that the module name is worth carrying at the call site,
# and the module names above make them reachable after `using PmSpectrum`:
#
#     SchrodingerFD.solve_fd, SchrodingerShooting.refine_spectrum,
#     TransferPrediction.predicted_eigenvalues, TransferPrediction.sd_from_halfwidth
#
# Each module exports its own API, so `using PmSpectrum.TransferPrediction` still
# brings those names in unqualified for code that uses them heavily.

end
