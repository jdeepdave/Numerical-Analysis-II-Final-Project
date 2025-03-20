using LinearAlgebra, SparseArrays, Plots, JLD2
plotly()
include("mesh.jl")

function u_0(x)
    return (x[1]^2 + x[2]^2)^2 - 1
end

function f(x)
    return 16 * (x[1]^2 + x[2]^2)
end

mesh_sizes = [2^(-3), 2^(-4), 2^(-5), 2^(-6)]
errors = []

for (i, hmax) in enumerate(mesh_sizes)
    mesh_file = "iso_mesh_$(i).jld2"

    if isfile(mesh_file)
        println("Loading precomputed isoparametric mesh for h = $hmax...")
        p, t = load_mesh(mesh_file)
    else
        println("Generating initial linear mesh for h = $hmax...")
        p, t = generate_initial_mesh(hmax)
        println("Converting to quadratic mesh for h = $hmax...")
        p, t = to_quadratic(p, t)
        println("Applying isoparametric transformation...")
        p, t = apply_isoparametric_transform(p, t)
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
        
        B = [y[2] - y[3] y[3] - y[1] y[1] - y[2] 0 0 0;
             x[3] - x[2] x[1] - x[3] x[2] - x[1] 0 0 0] / (2 * area)
        
        Ke = area * (B' * B)

        for a in 1:6
            for b in 1:6
                A[nodes[a], nodes[b]] += Ke[a, b]
            end
        end

        be = fill(area / 6, 6) .* f.([p[:, nodes[a]] for a in 1:6])
        for a in 1:6
            b[nodes[a]] += be[a]
        end
    end


    boundary_nodes = detect_boundary_nodes(t)
    println("Boundary nodes: ", boundary_nodes)
    for i in boundary_nodes
        A[i, :] .= 0
        A[i, i] = 1.0
        b[i] = 0.0
    end

    println("Solving Isoparametric FEM system for h = $hmax with $(n_nodes) nodes and $(n_elems) elements...")
    A += I * 1e-12
    uh = A \ b

    error = sqrt(sum((uh[i] - u_0(p[:, i]))^2 for i in 1:n_nodes) / n_nodes)
    push!(errors, error)

    println("Error for h = $hmax: $error")
end

p = plot(log.(mesh_sizes), log.(errors), marker=:o, xlabel="log(h)", ylabel="log(error)", title="Error Convergence (Isoparametric FEM)", lw=2)

savefig("isoparametric_error_convergence.png")
display(p)

println("Computation completed successfully! The plot should now appear.")

rates = []
for j in 1:length(mesh_sizes)-1
    rate = (log(errors[j+1]) - log(errors[j])) / (log(mesh_sizes[j+1]) - log(mesh_sizes[j]))
    push!(rates, rate)
end
println("Convergence rates: ", rates)

average_rate = sum(rates) / length(rates)
println("Average Convergence rate: $average_rate")

println("Isoparametric FEM Errors: ", errors)


