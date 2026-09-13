using Test
using PmSpectrum
using QuadGK: quadgk

@testset "PmSpectrum" begin
    @testset "branch inverse"          begin include("branch_inverse_test.jl")          end
    @testset "interval helpers"        begin include("interval_helpers_test.jl")        end
    @testset "B-spline basis"          begin include("bspline_basis_test.jl")           end
    @testset "density integration"     begin include("density_integration_test.jl")     end
    @testset "correlation curves"      begin include("correlation_test.jl")             end
    @testset "Galerkin matrices"       begin include("bspline_galerkin_matrix_test.jl") end
    @testset "results serialization"   begin include("results_test.jl")                 end
    @testset "perturbation solution"   begin include("construct_perturbation_test.jl")  end
    @testset "utils"                   begin include("utils_test.jl")                   end
    @testset "block matrix"            begin include("block_matrix_test.jl")            end
    @testset "Schrödinger FD"          begin include("schrodinger_fd_test.jl")          end
    @testset "Schrödinger shooting"    begin include("schrodinger_shooting_test.jl")    end
    @testset "transfer prediction"     begin include("transfer_prediction_test.jl")     end
    @testset "resolvent norm"          begin include("resolvent_norm_test.jl")          end
end
