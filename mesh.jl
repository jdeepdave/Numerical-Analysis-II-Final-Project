using JLD2
include("polytrimesh.jl")

function generate_initial_mesh(hmax)
    p, t = circlemesh(hmax)
    return p, t
end

function refine_mesh(p, t, num_refinements)
    for r in 1:num_refinements
        println("Refinement step $r: Before refinement → Nodes: ", size(p,2), ", Elements: ", size(t,2))
        
        p, t = retessellate(p, t)  # Refine the mesh
        boundary_nodes = detect_boundary_nodes(t)  # Get boundary nodes

        # Project boundary nodes back to the unit circle
        for i in boundary_nodes
            r = sqrt(p[1,i]^2 + p[2,i]^2)
            if r > 0
                p[:, i] .= p[:, i] ./ r
            end
        end

        println("After refinement → Nodes: ", size(p,2), ", Elements: ", size(t,2))
    end
    return p, t
end



function save_mesh(p, t, filename)
    jldsave(filename; p, t)
end

function load_mesh(filename)
    data = load(filename)
    return data["p"], data["t"]
end
