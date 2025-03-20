using LinearAlgebra, SparseArrays, Plots, JLD2
plotly()
include("GaussianIntegraton.jl") 
include("mesh.jl") 

function u_T(x)
    return (x[1]^2 + x[2]^2)^2 - 1
end

mesh_sizes = [2^(-3), 2^(-4), 2^(-5), 2^(-6)]
errors = []

for (i, hmax) in enumerate(mesh_sizes)
    println("Generating initial linear mesh for h = $hmax...")
    p, t = generate_initial_mesh(hmax)

    println("Converting to quadratic mesh for h = $hmax...")
    p, t = to_quadratic(p, t)

    n_nodes = size(p, 2)
    n_elems = size(t, 2)

    A = spzeros(n_nodes, n_nodes)
    b = zeros(n_nodes)

    println("Assembling system using Gaussian quadrature...")

    for i in 1:n_elems
        nodes = t[:, i]
        x = p[:, nodes]  
        if size(x, 2) != 6
            error("Mesh element does not have six nodes, check `to_quadratic()` conversion.")
        end

        J = [x[:, 2] - x[:, 1] x[:, 3] - x[:, 1]]
        detJ = abs(det(J))

        Ke = zeros(6, 6) 
        be = zeros(6)

        gc_reshaped = gc' 
        gc_reshaped = gc_reshaped[1:2, :] 

        for q in 1:size(gc_reshaped, 2)
            ξ, η = gc_reshaped[:, q] 
            w = gw[q] 

            ∇N_ref = [
                4*ξ - 1   0   4*η  4*(1 - ξ - η) -4*ξ  -4*η;
                0   4*η - 1   4*ξ  -4*ξ  -4*η   4*(1 - ξ - η)
            ]

            ∇N_phys = J' \ ∇N_ref

            Ke += (∇N_phys' * ∇N_phys) * w * detJ

            be += fill(16 * (x[1,1]^2 + x[2,1]^2) * w * detJ, 6)
        end

        for a in 1:6
            for b in 1:6
                A[nodes[a], nodes[b]] += Ke[a, b]
            end
            b[nodes[a]] += be[a]
        end
    end

    boundary_nodes = detect_boundary_nodes(t)
    for i in boundary_nodes
        A[i, :] .= 0
        A[i, i] = 1.0
        b[i] = 0.0
    end

    println("Solving FEM system for h = $hmax with $(n_nodes) nodes and $(n_elems) elements...")
    uh = A \ b

    error = sqrt(sum((uh[i] - u_T(p[:, i]))^2 for i in 1:n_nodes) / n_nodes)
    push!(errors, error)

    println("Error for h = $hmax: $error")
end

p = plot(log.(mesh_sizes), log.(errors), marker=:o, xlabel="log(h)", ylabel="log(error)", title="Error Convergence with Gaussian Quadrature", lw=2)

println("Computation completed successfully!")
println("Quadratic FEM Errors: ", errors)

rates = []
for j in 1:length(mesh_sizes)-1
    rate = (log(errors[j+1]) - log(errors[j])) / (log(mesh_sizes[j+1]) - log(mesh_sizes[j]))
    push!(rates, rate)
end
println("Convergence rates: ", rates)

average_rate = sum(rates) / length(rates)
println("Average Convergence Rate: $average_rate")

savefig("quadratic_error_convergence.png")
display(p)