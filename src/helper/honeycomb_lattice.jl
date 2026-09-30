# 1/4/2024
# C-style honeycomb lattice for the Kitaev cluster: the LatticeBond / WedgeBond
# structs, the bond builder, and the three-spin wedge builder.

"""
    LatticeBond

A single bond of a lattice or interaction graph: site numbers `s1`, `s2`,
optional endpoint coordinates `(x1, y1)`, `(x2, y2)`, and a free-form `type`
string.
"""
struct LatticeBond
  s1::Int
  s2::Int
  x1::Float64
  y1::Float64
  x2::Float64
  y2::Float64
  type::String
end

"""
    LatticeBond(s1::Int, s2::Int)
    LatticeBond(s1::Int, s2::Int, x1::Real, y1::Real,
                x2::Real, y2::Real, type::String = "")

Construct a `LatticeBond` from the two site numbers alone (coordinates zeroed,
empty type), or with explicit endpoint coordinates and an optional type string.
"""
function LatticeBond(s1::Int, s2::Int)
  return LatticeBond(s1, s2, 0.0, 0.0, 0.0, 0.0, "")
end

function LatticeBond(
  s1::Int, s2::Int, x1::Real, y1::Real, x2::Real, y2::Real, bondtype::String=""
)
  cf(x) = convert(Float64, x)
  return LatticeBond(s1, s2, cf(x1), cf(y1), cf(x2), cf(y2), bondtype)
end

"""
    Lattice

Alias for `Vector{LatticeBond}`.
"""
const Lattice = Vector{LatticeBond}



"""
    honeycomb_lattice_Cstyle(Nx::Int, Ny::Int; yperiodic=false)::Lattice

Bond list of the `Nx × Ny` honeycomb cluster in the C-style ordering scheme
(`Nx` needs to be an even number); `yperiodic = true` (requires `Ny > 2`)
adds the y-wrap bonds of the cylinder.
"""
function honeycomb_lattice_Cstyle(Nx::Int, Ny::Int; yperiodic=false)::Lattice
	yperiodic = yperiodic && (Ny > 2)
	N = Nx * Ny
	Nbond = trunc(Int, 3/2 * N) - Ny + (yperiodic ? 0 : -trunc(Int, Nx / 2))
	# @show Nbond

	latt = Lattice(undef, Nbond)
	b = 0
	for n in 1:N
		tmp = div(n - 1, 2 * Ny)
		x = 2 * tmp + mod(n - 1, 2) + 1
		y = mod(div(n - 1, 2), Ny) + 1
		# @show n, x, y

		# x-direction bonds for A sublattice
		if mod(x, 2) == 0 && x < Nx
			latt[b += 1] = LatticeBond(n, n + 2 * Ny - 1)
			# @show n, x, y, b
		end

		# bonds for B sublattice
		if Ny > 1
			if mod(x, 2) == 1 && x < Nx
				# @show latt
				latt[b += 1] = LatticeBond(n, n + 1)
				if y != 1
					latt[b += 1] = LatticeBond(n, n - 1)
				end
			end

			# periodic bonds
			if mod(x, 2) == 1 && yperiodic && y == 1
				latt[b += 1] = LatticeBond(n, n + 2 * Ny - 1)
			end
		end
	end
	# @show latt
	return latt
end



"""
    WedgeBond

A three-site wedge — two bonds sharing the center site `s2` — carrying a
three-spin interaction term: site numbers `s1`, `s2`, `s3`, optional site
coordinates, and a free-form `type` string.
"""
struct WedgeBond
  s1::Int
  s2::Int
  s3::Int
  x1::Float64
  y1::Float64
  x2::Float64
  y2::Float64
  x3::Float64
  y3::Float64
  type::String
end


function WedgeBond(s1::Int, s2::Int, s3::Int)
	return WedgeBond(s1, s2, s3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, "")
end


function WedgeBond(
  s1::Int, s2::Int, s3::Int, x1::Real, y1::Real, x2::Real, y2::Real, x3::Real, y3::Real, bondtype::String=""
)
  cf(x) = convert(Float64, x)
  return WedgeBond(s1, s2, s3, cf(x1), cf(y1), cf(x2), cf(y2), cf(x3), cf(y3), bondtype)
end



"""
    honeycomb_Cstyle_wedge(Nx::Int, Ny::Int; yperiodic=false)

Vector of `WedgeBond`s (three-spin terms) for the `Nx × Ny` C-style honeycomb
cluster: one wedge per pair of distinct bonds sharing the center site `s2`.
The x-boundary is always open. With `yperiodic = false` the wedges that would
pass through the missing y-wrap bond are dropped (two per y-boundary site
except the two degree-1 corners), giving `3N - 4Ny - (2Nx - 2)` wedges; with
`yperiodic = true` (requires `Ny > 2`) the wrap wedges are kept and the count
is `3N - 4Ny`.
"""
function honeycomb_Cstyle_wedge(Nx::Int, Ny::Int; yperiodic=false)
	yperiodic = yperiodic && (Ny > 2)
	N = Nx * Ny									# Number of sites on the lattice
	Nwedge = 3 * N - 4 * Ny - (yperiodic ? 0 : 2 * Nx - 2)
	wedge = Vector{WedgeBond}()
	sizehint!(wedge, Nwedge)

	# Set up the three-spin interaction terms
	for n in 1 : N
		# Calculate the (x, y) coordinates of the site n based on C-style ordering
		tmp = div(n - 1, 2 * Ny)
		x = 2 * tmp + mod(n - 1, 2) + 1
		y = mod(div(n - 1, 2), Ny) + 1


		# Set up the three-spin interaction terms based on the (x, y) coordinates of the site n
		if isodd(x)
			# Wedge between the two zigzag neighbours (x + 1, y) and (x + 1, y - 1)
			if y > 1
				push!(wedge, WedgeBond(n - 1, n, n + 1))
			elseif yperiodic
				push!(wedge, WedgeBond(n + 1, n, n + 2 * Ny - 1))
			end

			# Wedges pairing the horizontal neighbour (x - 1, y) with each zigzag neighbour
			if x != 1
				push!(wedge, WedgeBond(n - 2 * Ny + 1, n, n + 1))
				if y > 1
					push!(wedge, WedgeBond(n - 2 * Ny + 1, n, n - 1))
				elseif yperiodic
					push!(wedge, WedgeBond(n - 2 * Ny + 1, n, n + 2 * Ny - 1))
				end
			end
		else
			# Wedge between the two zigzag neighbours (x - 1, y) and (x - 1, y + 1)
			if y < Ny
				push!(wedge, WedgeBond(n - 1, n, n + 1))
			elseif yperiodic
				push!(wedge, WedgeBond(n - 2 * Ny + 1, n, n - 1))
			end

			# Wedges pairing the horizontal neighbour (x + 1, y) with each zigzag neighbour
			if x != Nx
				push!(wedge, WedgeBond(n - 1, n, n + 2 * Ny - 1))
				if y < Ny
					push!(wedge, WedgeBond(n + 1, n, n + 2 * Ny - 1))
				elseif yperiodic
					push!(wedge, WedgeBond(n - 2 * Ny + 1, n, n + 2 * Ny - 1))
				end
			end
		end
	end


	# Check if the number of wedges generated matches the expected number
	if length(wedge) != Nwedge
		error("Generated $(length(wedge)) wedges but expected $Nwedge.")
	end


	# Display the three-spin interaction terms for debugging purposes
	# for (_, w) in enumerate(wedge)
	# 	@show w.s1, w.s2, w.s3
	# end
	# @show wedge

	return wedge
end
