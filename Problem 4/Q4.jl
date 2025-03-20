using LinearAlgebra, SparseArrays, Arpack, Plots, JLD2
plotly()
include("mesh.jl")  
include("GaussianIntegraton.jl") 

λ_true = 5.78318596294678499703

mesh_sizes = [2^(-3), 2^(-4), 2^(-5), 2^(-6)]
eigenvalues = []

function apply_isoparametric_transform(p, t)
    boundary_nodes = detect_boundary_nodes(t)
    for i in boundary_nodes
        r = sqrt(p[1, i]^2 + p[2, i]^2)
        if r > 0
            p[:, i] .= p[:, i] ./ r 
        end
    end
    return p, t
end

for (i, hmax) in enumerate(mesh_sizes)
    println("🔹 Generating initial linear mesh for h = $hmax...")
    p, t = generate_initial_mesh(hmax)

    println("🔹 Converting to quadratic mesh for h = $hmax...")
    p, t = to_quadratic(p, t) 

    println("🔹 Applying isoparametric transformation...")
    p, t = apply_isoparametric_transform(p, t)

    n_nodes = size(p, 2)
    n_elems = size(t, 2)

    A = spzeros(n_nodes, n_nodes)
    M = spzeros(n_nodes, n_nodes)

    println("🔹 Assembling FEM system for eigenvalue computation...")

    for i in 1:n_elems
        nodes = t[:, i]
        x = p[:, nodes]

        J = [x[:, 2] - x[:, 1] x[:, 3] - x[:, 1]]
        detJ = abs(det(J))

        Ke = zeros(6, 6)
        Me = zeros(6, 6)

        for q in 1:size(gc, 2)
            ξ, η = gc[1:2, q] 
            w = gw[q] 

            ∇N_ref = [
                4*ξ - 1   0   4*η  4*(1 - ξ - η) -4*ξ  -4*η;
                0   4*η - 1   4*ξ  -4*ξ  -4*η   4*(1 - ξ - η)
            ]

            ∇N_phys = J' \ ∇N_ref

            Ke += (∇N_phys' * ∇N_phys) * w * detJ
            Me += detJ * w * I(6) / 6 
        end

        for a in 1:6
            for b in 1:6
                A[nodes[a], nodes[b]] += Ke[a, b]
                M[nodes[a], nodes[b]] += Me[a, b]
            end
        end
    end

    boundary_nodes = detect_boundary_nodes(t)
    for i in boundary_nodes
        A[i, :] .= 0
        A[:, i] .= 0
        A[i, i] = 1.0
        M[i, :] .= 0
        M[:, i] .= 0
        M[i, i] = maximum(diag(M)) 
    end

    println("Matrix A rank: ", rank(A))
    println("Matrix M rank: ", rank(M))
    println("Condition number of A: ", cond(Matrix(A)))
    println("Condition number of M: ", cond(Matrix(M)))

    println("🔹 Solving eigenvalue problem for h = $hmax...")
    λ, _ = eigs(A, M, nev=3, which=:SM, ncv=40, maxiter=100_000) 

    λ_min = filter(x -> 5.0 < x < 6.0, real(λ))[1]
    push!(eigenvalues, λ_min)
    println("Computed eigenvalue for h = $hmax: ", λ_min)
end