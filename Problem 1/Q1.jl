using LinearAlgebra, SparseArrays, Plots, JLD2
plotly()
include("mesh.jl")

function u_T(x)
    return 10 * (x[1]^2 + x[2]^2 - 1)
end

mesh_sizes = [2^(-3), 2^(-4), 2^(-5), 2^(-6)]
errors = []

for (i, hmax) in enumerate(mesh_sizes)
    mesh_file = "mesh_$(i).jld2"

    if isfile(mesh_file)
        p, t = load_mesh(mesh_file)
    else
        p, t = generate_initial_mesh(hmax) 

        p, t = refine_mesh(p, t, 2)

        save_mesh(p, t, mesh_file) 
    end

    n_nodes = size(p, 2)
    n_elems = size(t, 2)

    A = spzeros(n_nodes, n_nodes)
    b = zeros(n_nodes)

    for i in 1:n_elems
        nodes = t[:, i]
        x = p[1, nodes]
        y = p[2, nodes]

        area = 0.5 * abs(x[1]*(y[2] - y[3]) + x[2]*(y[3] - y[1]) + x[3]*(y[1] - y[2]))
        
        B = [y[2] - y[3] y[3] - y[1] y[1] - y[2];
             x[3] - x[2] x[1] - x[3] x[2] - x[1]] / (2 * area)
        
        Ke = area * (B' * B)

        for a in 1:3
            for b in 1:3
                A[nodes[a], nodes[b]] += Ke[a, b]
            end
        end
        
        be = fill(40 * area / 3, 3)
        for a in 1:3
            b[nodes[a]] += be[a]
        end
    end

    boundary_nodes = detect_boundary_nodes(t)
    for i in boundary_nodes
        r = sqrt(p[1,i]^2 + p[2,i]^2)
        if r > 0
            p[:, i] .= p[:, i] ./ r
        end
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

p = plot(log.(mesh_sizes), log.(errors), marker=:o, xlabel="log(h)", ylabel="log(error)", title="Error Convergence", lw=2)

savefig("error_convergence.png")

display(p)



println("Computation completed successfully! The plot should now appear.")

rates = []
for j in 1:length(mesh_sizes)-1
    rate = (log(errors[j+1]) - log(errors[j])) / (log(mesh_sizes[j+1]) - log(mesh_sizes[j]))
    push!(rates, rate)
end
println("Convergence rates: ", rates)

average_rate = sum(rates) / length(rates)
println("Average Convergence Rate: $average_rate")

println("Problem 1 Errors: ", errors)