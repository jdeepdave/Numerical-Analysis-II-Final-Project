# -------------------------------------------------------------------------------
# polytrimesh.jl
#
# Simple wrapper around the Triangulate package for quality meshing of
# polygons with an optional size function.
#
# Per-Olof Persson, UC Berkeley <persson@berkeley.edu>, 2025
# -------------------------------------------------------------------------------

using Triangulate

"""
    p,t = polytrimesh(pvs, holes, hmax, cmd)

Unstructured meshing of polygonal regions.
    `pvs`: Array of polygons (2xN arrays)
    `holes`: x,y coordinates of holes in geometry (2xM array)
    `hmax`: Scalar or size function (default Inf)
    `cmd`: Command to triangle mesh generator (string, default "puq28.6Q")

Example:
```julia
pv1 = hcat([0,0], [1,0], [1,1], [.5,.5], [0,1], [0,0])
pv2 = hcat([.2,.2], [.4,.2], [.4,.4], [.39,.4], [.2,.4], [.2,.2])
holes = [.3,.3]
hmaxfcn = (x,y) -> 0.01 + 0.3*abs(sqrt((x-.5)^2+(y-.5)^2))
p,t = polytrimesh([pv1,pv2], holes, hmaxfcn)
# Optional plotting:
using Plots, TriplotRecipes
trimesh(p[1,:], p[2,:], t, aspect_ratio=:equal)
```
"""
function polytrimesh(pvs, holes=zeros(2,0), hmax=Inf, cmd="puq28.6Q")
  pv = Matrix{Float64}[]
  seg = Matrix{Int32}[]
  nseg = 0
  for cpv in pvs
    closed = false
    if size(cpv,2) > 1 & isapprox(cpv[:,1], cpv[:,end])
      closed = true
      cpv = cpv[:,1:end-1]
    end
    np = size(cpv,2)

    if np > 1
      cseg = [mod(i+j-1,np)+1 for i = 0:1, j = 1:np]
      if !closed
        cseg = cseg[:,1:end-1]
      end

      push!(seg, cseg .+ nseg)
      nseg += size(cseg,2)
    end

    push!(pv, cpv)
  end

  pv = hcat(pv...)
  seg = hcat(seg...)
  triin = Triangulate.TriangulateIO()
  triin.pointlist = pv
  triin.segmentlist = seg
  triin.segmentmarkerlist=Vector{Int32}(1:size(seg,2))
  if holes != nothing
    triin.holelist = reshape(holes, 2, :)
  end

  function unsuitable(x1,y1,x2,y2,x3,y3,area)
    if isa(hmax, Number)
      sz = hmax
    else
      sz = hmax((x1+x2+x3)/3, (y1+y2+y3)/3)
    end
    elemsz = sqrt(maximum([(x1-x2)^2+(y1-y2)^2,
                           (x2-x3)^2+(y2-y3)^2,
                           (x3-x1)^2+(y3-y1)^2]))
    elemsz > sz
  end
  triunsuitable(unsuitable)

  (triout, vorout)=triangulate(cmd, triin)
  triout.pointlist, triout.trianglelist
end

# -------------------------------------------------------------------------------
# Sample meshes

function samplemesh1()
  pv1 = hcat([0,0], [1,0], [1,1], [.5,.5], [0,1], [0,0])
  pv2 = hcat([.2,.2], [.4,.2], [.4,.4], [.39,.4], [.2,.4], [.2,.2])
  holes = [.3,.3]
  hmaxfcn = (x,y) -> 0.01 + 0.3*abs(sqrt((x-.5)^2+(y-.5)^2))
  p,t = polytrimesh([pv1,pv2], holes, hmaxfcn)
  p,t
end

function circlemesh(hmax)
  n = ceil(Int, 2π / hmax)
  θ = 2π*(0:n)'./n
  pv = vcat(cos.(θ), sin.(θ))
  p,t = polytrimesh([pv], [], hmax, "puYQ")
end

function detect_boundary_nodes(t)
  # Initialize a dictionary to count occurrences of edges
  edge_count = Dict{Array{Int32, 1}, Int}()

  # Loop over each triangle in t (3xM matrix in Julia)
  for i in 1:size(t, 2)
      # Extract the nodes of the triangle (3 nodes per triangle)
      T = t[:, i]

      # Define the 3 edges of the triangle as sorted tuples
      
      if length(T) == 3
        edge1 = sort([T[1], T[2]])
        edge2 = sort([T[2], T[3]])
        edge3 = sort([T[3], T[1]])

        # Count the occurrences of the edges
        edge_count[edge1] = get(edge_count, edge1, 0) + 1
        edge_count[edge2] = get(edge_count, edge2, 0) + 1
        edge_count[edge3] = get(edge_count, edge3, 0) + 1
      elseif length(T) == 6
        edge1 = sort([T[1], T[4]])
        edge2 = sort([T[4], T[2]])
        edge3 = sort([T[2], T[5]])
        edge4 = sort([T[5], T[3]])
        edge5 = sort([T[3], T[6]])
        edge6 = sort([T[6], T[1]])

        # Count the occurrences of the edges
        edge_count[edge1] = get(edge_count, edge1, 0) + 1
        edge_count[edge2] = get(edge_count, edge2, 0) + 1
        edge_count[edge3] = get(edge_count, edge3, 0) + 1
        edge_count[edge4] = get(edge_count, edge4, 0) + 1
        edge_count[edge5] = get(edge_count, edge5, 0) + 1
        edge_count[edge6] = get(edge_count, edge6, 0) + 1

      end
  end

  # Initialize a set to collect unique boundary edges
  boundary_edges_set = Vector{Vector{Int32}}()

  # Loop over the edges and collect the boundary edges (those that appear exactly once)
  for (edge, count) in edge_count
      if count == 1  # Boundary edges appear exactly once
          push!(boundary_edges_set, edge)
      end
  end

  # Convert the set to an array before returning it
  all = vcat([tuple[i] for tuple in boundary_edges_set for i in 1:2]...)

  return unique(all)

end

function elim_unused_nodes(p::Matrix{Float64}, t::Matrix{Int32})
  P = size(p, 2)  # Number of points (columns in Julia)
  T1, T2 = size(t)  # Size of the t matrix

  # Identifying and deleting unused rows
  used_nodes = unique(t)
  unused_nodes = setdiff(1:P, sort(used_nodes))

  # Removing unused nodes from p
  p = p[:, setdiff(1:P, unused_nodes)]

  # Reorganizing t
  whatToSubtract = zeros(Int32, P)
  current = 0
  placeInUNR = 1
  unused_nodes = sort(unused_nodes)

  for i in 1:P
      if placeInUNR <= length(unused_nodes) && i == unused_nodes[placeInUNR]
          current += 1
          placeInUNR += 1
      end
      whatToSubtract[i] = current
  end

  # Adjusting t
  t = [t[r, c] - whatToSubtract[t[r, c]] for r in 1:T1, c in 1:T2]

  return p, t
end

function replace_values(t::Matrix{Int32}, old_values::Vector{Int32}, new_values::Vector{Int32})
  mapping = Dict(old_values .=> new_values)
  return reshape([get(mapping, x, x) for x in t], size(t))
end


function elim_dup_nodes(p::Matrix{Float64}, t::Matrix{Int32}, tol::Float64 = 1e-10)
  P = size(p, 2)  # Number of points (columns in Julia)

  # Sort rows and get indices
  sorted_indices = sortperm(eachcol(p), by=x -> x)
  C = p[:, sorted_indices]
  IX = sorted_indices

  tbr = zeros(Int32, P)
  tbrw = zeros(Int32, P)
  count = 0
  r = 1

  while r <= P - 1
      r1 = r + 1
      if r1 > size(C, 2)
          break
      end
      while norm(C[:, r] - C[:, r1]) < tol
          if r1 > size(C, 2)
              break
          end
          count += 1
          tbrw[count] = min(IX[r1], IX[r])
          tbr[count] = max(IX[r1], IX[r])
          r1 += 1
      end
      r = r1
  end

  # Replace tbr values in t with tbrw
  t = replace_values(t, tbr[1:count], tbrw[1:count])

  p, t = elim_unused_nodes(p, t)

  return p, t
end


function retessellate(p,t)
  
  P = size(p, 2);
  T = size(t, 2);

  newT = zeros(Int32, 3, 4*T);

  count = 1;
  p = hcat(p, zeros(2, 3*T))

  lastT = 0

  M = [
    .5 .0 .5
    .5 .5 .0
    .0 .5 .5
  ]

  for i = 1:T
      p[:, [ P+1 P+2 P+3]] = p[:,t[:,i]]*M
      
      newT[:, lastT + 1] = [ t[1, i] P+1 P+3 ];
      newT[:, lastT + 2] = [ t[2, i] P+2 P+1 ];
      newT[:, lastT + 3] = [ t[3, i] P+3 P+2 ];
      newT[:, lastT + 4] = [ P+1 P+2 P+3 ];

      lastT += 4
      P += 3
  end

  return elim_dup_nodes(p, newT)
end

function to_quadratic(p,t)
  
  P = size(p, 2);
  T = size(t, 2);

  newT = zeros(Int32, 3, 4*T);

  count = 1;
  p = hcat(p, zeros(2, 3*T))

  lastT = 0

  t = vcat(t, zeros(Int32, 3, T))

  for i=1:T
      p[:, P + 1] = .5*(p[:, t[1,i]] + p[:, t[2,i]])
      p[:, P + 2] = .5*(p[:, t[2,i]] + p[:, t[3,i]])
      p[:, P + 3] = .5*(p[:, t[3,i]] + p[:, t[1,i]])
      
      t[[ 4 5 6 ], i] = P .+ [ 1 2 3 ]

      P += 3
  end

  return elim_dup_nodes(p, t)
end

function detect_edges(p, t)
  # Initialize an array to store the edges for each triangle
  edges_list = []

  # Loop over each triangle in t (3xM matrix in Julia)
  for i in 1:size(t, 2)
      # Extract the nodes of the triangle (3 nodes per triangle)
      triangle = t[:, i]

      # Define the 3 edges of the triangle as sorted tuples
      edge1 = sort([Int32(triangle[1]), Int32(triangle[2])])
      edge2 = sort([Int32(triangle[2]), Int32(triangle[3])])
      edge3 = sort([Int32(triangle[3]), Int32(triangle[1])])

      # Store the edges as an array in the edges_list
      push!(edges_list, [edge1, edge2, edge3])
  end

  # Return the list of edges
  return edges_list
end


