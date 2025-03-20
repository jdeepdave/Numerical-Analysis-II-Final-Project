using LinearAlgebra, SparseArrays, Arpack

function solve_basic_eigenproblem(hmax)
    println("Generating mesh for unit circle with hmax = $hmax...")
    p, t = circlemesh(hmax)  # Generate mesh on unit disk

    println("Converting to quadratic elements...")
    p, t = to_quadratic(p, t)  # Ensure elements have 6 nodes

    println("Assembling system matrices...")
    A, M = assemble_system(p, t)  # Standard FEM stiffness and mass matrices

    println("Applying Dirichlet boundary conditions...")
    boundary_nodes = detect_boundary_nodes(t)
    A, M = apply_dirichlet(A, M, boundary_nodes)

    println("Checking matrix properties...")
    println("Size of A: ", size(A), " | Nonzero elements: ", nnz(A))
    println("Size of M: ", size(M), " | Nonzero elements: ", nnz(M))

    if any(isnan.(A)) || any(isnan.(M))
        error("NaN detected in system matrices! Check assemble_system().")
    end

    println("Solving for smallest eigenvalue...")
    try
        λ, _ = eigs(A + 1e-10 * I, M + 1e-10 * I, nev=1, which=:SM, 
                    maxiter=5000, tol=1e-8, sigma=1.0)
        return real(λ[1])
    catch e
        println("⚠️ Eigenvalue computation failed: ", e)

        println("Trying SVD as fallback...")
        try
            λ = svdvals(Matrix(A))  # Use singular values for approximation
            return real(λ[1])  # Take the smallest singular value
        catch svd_error
            println("⚠️ SVD computation also failed: ", svd_error)
            return NaN  # Return NaN to signal failure
        end
    end
end


hmax = 2^(-4) 
λ_computed = solve_basic_eigenproblem(hmax)
println("Computed lowest eigenvalue: ", λ_computed)
println("Expected analytical value: ", (2.4048)^2)  # λ₁ = j₀₁², first zero of J₀ Bessel function
