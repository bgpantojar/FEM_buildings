using Amaru

function main(args)
	mesh = Mesh(args[1], reorder=true)
	tag!(mesh.elems[:solids], "solids")
	# Model definition
	materials = [
	    "solids" => ElasticSolid(E = parse(Float64, args[2]), nu = parse(Float64, args[3]), rho = parse(Float64, args[4]))
	]

	model = Model(mesh, materials)

	# Finite element modeling
	bcs = [
	    :(z==0)   => NodeBC(ux=0, uy=0, uz=0),
	]

	mod_solve!(model, bcs, nmods=parse(Int64, args[5]), outdir=args[6])
end
main(ARGS)