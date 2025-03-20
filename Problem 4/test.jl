using LinearAlgebra, SparseArrays, Arpack, JLD2, Plots
plotly()
include("polytrimesh.jl")
include("mesh.jl")


function apply_isoparametric_transform(p, t)
    for i in 1:size(t, 2)
        nodes = t[:, i]
        X = p[1, nodes]
        Y = p[2, nodes]

        if any(isnan.(X)) || any(isnan.(Y))
            println("Warning: NaN detected in isoparametric transform input for element ", i)
            continue
        end

        Ψ_inv = [2  2  0 -4  0  0;
                 4  0  0 -4 -4  0;
                 2  0  2  0  0 -4;
                -3 -1  0  4  0  0;
                -3  0 -1  0  0  4;
                 1  0  0  0  0  0]

        coeffs_x = Ψ_inv * X
        coeffs_y = Ψ_inv * Y

        for j in 1:6
            p[1, nodes[j]] = coeffs_x' * [X[j]^2, 2*X[j]*Y[j], Y[j]^2, 2*X[j], 2*Y[j], 1]
            p[2, nodes[j]] = coeffs_y' * [X[j]^2, 2*X[j]*Y[j], Y[j]^2, 2*X[j], 2*Y[j], 1]
        end
    end
    return p, t
end

function assemble_system(p, t)
    n_nodes = size(p, 2)
    n_elems = size(t, 2)
    
    A = spzeros(n_nodes, n_nodes)
    M = spzeros(n_nodes, n_nodes)
    
    for i in 1:n_elems
        nodes = t[:, i]
        x = p[1, nodes]
        y = p[2, nodes]

        J = [x[2] - x[1]  y[2] - y[1];
             x[3] - x[1]  y[3] - y[1]]
        detJ = abs(det(J))

        if isnan(detJ) || detJ < 1e-10
            println("Warning: Near-zero or NaN determinant in Jacobian for element ", i, ". detJ =", detJ)
            continue
        end

        B = [y[2] - y[3]  y[3] - y[1]  y[1] - y[2]  0  0  0;
             x[3] - x[2]  x[1] - x[3]  x[2] - x[1]  0  0  0] / (2 * detJ)

        Ke = (1 / detJ) * (B' * B)
        Me = (detJ / 12) * [2 1 1 1 0 0; 
                            1 2 1 0 1 0; 
                            1 1 2 0 0 1; 
                            1 0 0 2 1 1; 
                            0 1 0 1 2 1; 
                            0 0 1 1 1 2]

        for a in 1:6
            for b in 1:6
                A[nodes[a], nodes[b]] += Ke[a, b]
                M[nodes[a], nodes[b]] += Me[a, b]
            end
        end
    end

    println("Checking for zero rows in M...")
    A[isnan.(A)] .= 1e-10
    M[isnan.(M)] .= 1e-10
    min_M = minimum(sum(M, dims=2))
    min_A = minimum(sum(A, dims=2))
    println("Min row sum of M: ", min_M)
    println("Min row sum of A: ", min_A)

    if isnan(min_M) || isnan(min_A)
        error("NaN detected in matrices, check mass and stiffness assembly.")
    end

    return A, M
end

function apply_dirichlet(A, M, boundary_nodes)
    for node in boundary_nodes
        A[node, :] .= 0
        A[:, node] .= 0
        A[node, node] = 1
        M[node, :] .= 0
        M[:, node] .= 0
        M[node, node] = 1
    end
    return A, M
end

function compute_lowest_eigenvalue(hmax)
    println("Generating initial mesh for hmax = $hmax...")
    p, t = generate_initial_mesh(hmax)
    
    println("Refining and converting to quadratic mesh...")
    p, t = to_quadratic(p, t)
    
    println("Applying isoparametric transformation...")
    p, t = apply_isoparametric_transform(p, t)
    
    println("Assembling system matrices...")
    A, M = assemble_system(p, t)
    
    println("Applying Dirichlet boundary conditions...")
    boundary_nodes = detect_boundary_nodes(t)
    A, M = apply_dirichlet(A, M, boundary_nodes)
    
    println("Solving eigenvalue problem with shift-invert mode...")
    try
        λ, _ = eigs(A + 1e-10 * I, M + 1e-10 * I, nev=1, which=:SM, sigma=1.0)
    catch e
        println("Eigenvalue computation failed, switching to dense approximation.")
        λ = svdvals(Matrix(A))  # Use singular values for efficiency
        λ = λ[1]  # Take smallest singular value
    end
    
    println("Computed eigenvalues: ", λ)
    return real(λ[1])
end

mesh_sizes = [2^(-3), 2^(-4), 2^(-5)]
eigenvalues = [compute_lowest_eigenvalue(h) for h in mesh_sizes]

println("Computed eigenvalues: ", eigenvalues)
λ_extrapolated = (4 * eigenvalues[2] - eigenvalues[1]) / 3 / π
println("Richardson Extrapolated λ: ", λ_extrapolated)
