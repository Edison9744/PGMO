
using JuMP
using MathOptComplements
using NLPModels
using NLPModelsJuMP
using MadNLP
using CCOpt
using Random
using QuasiMonteCarlo 



include("examples/edf-example-1/quad_jump.jl")

# Load model
function load_model()
    model = quad_model(
        joinpath(@__DIR__, "examples/edf-example-1/optimTarif/run_tests/results/quad_b50_S50.dat"),
    )
    return model
end

function convert_jump_to_mpcc(model)
    # Convert to MPCCModel and pass the model to CCOpt
    MathOptComplements.Bridges.add_all_bridges(model)
    JuMP.set_silent(model)
    set_optimizer(model, CCOpt.Optimizer)
    JuMP.set_optimizer_attribute(model, "max_iter", 0)
    JuMP.optimize!(model)
    # Get MPCC formulation
    mpcc = JuMP.unsafe_backend(model).mpcc
    return mpcc
end

function build_madnlp(mpcc)
    # Convert complementarity constraints 0 <= x1 ⟂ x2 >= 0 as x1 x2 <= 0
    nlp = CCOpt.ScholtesRelaxation(mpcc)
    # Build MadNLP instance
    solver = MadNLP.MadNLPSolver(nlp)
    # Initialize data structure
    MadNLP.initialize!(solver)
    solver.status = MadNLP.REGULAR
    return solver
end

function generate_sobol_candidates(dim::Int, n_samples::Int; ub_default::Float64 = 1000.0)
    lb = zeros(dim)
    ub = fill(ub_default, dim)

    # Tirage de la matrice Sobol (taille: dim x n_samples)
    X = QuasiMonteCarlo.sample(n_samples, lb, ub, SobolSample())

    # Conversion en liste de Vecteur
    return [Vector{Float64}(X[:, k]) for k in 1:size(X, 2)]
end

function obj(nlp, x)

    list = Dict()
    for i in x
        f = NLPModels.obj(nlp, i)
        println("obj ", f," et x ", length(i))
        list[i] = f
    end

    #ordre décroissant
    x_trie = sort(collect(keys(list)), by = k -> list[k], rev = true)
    return x_trie
end


function max_compare(nlp, list)
    x_trie = obj(nlp, list)
    println("max_compare ", length(x_trie[1]) )  
    return x_trie[1]
end


function recherche_local(nlp, x, max_iter = 50, tol = 1e-6)
    nlp.meta.x0 .= x
    n = nlp.meta.nvar  # Nombre de variables d'origine (dimension de x0)

    solver = MadNLPSolver(
        nlp; 
        max_iter = max_iter,   # Limite le nombre d'itérations
        tol = tol,             # Tolérance d'arrêt
        print_level = MadNLP.ERROR # Désactive la plupart des affichages (optionnel)
    )
    solve!(solver)
    
    return copy(solver.x.values[1:n])    
end


function cherche(nlp, candidat_garde)

    list_max = Vector{Vector{Float64}}()
    list_new_vect = Vector{Vector{Float64}}()

    y_samp = candidat_garde[1]

    push!(list_new_vect, candidat_garde[1])
    push!(list_max, candidat_garde[1])

    # evolution de lambda
    kmax = length(candidat_garde)

    for r in 1:kmax
        x_sample = max_compare(nlp, list_max)

        if r == 1
        else 
            # evolution de lambda
            λ = 0.25 + 0.75 * (r - 1) / max(1, kmax - 1)
            y_samp = (1 - λ) .* candidat_garde[r] .+ λ .* x_sample
        end

        push!(list_new_vect, y_samp)
        push!(list_max, recherche_local(nlp, y_samp))
    end

    return list_max
end

function tiktak(mpcc)
    
    solver = build_madnlp(mpcc)
    nlp = solver.cb.nlp
    # Get initial primal position
    x0 = NLPModels.get_x0(nlp)

    # Get indices
    ind_x1 = mpcc.meta.ind_cc1
    ind_x2 = mpcc.meta.ind_cc2
    # tiktak

    N = 200
    N_garde = 50

    candidat = generate_sobol_candidates(length(x0), N)
    candidat_trie = obj(nlp, candidat) 

    candidat_garde = candidat_trie[1:min(N_garde, end)] 

    best_x = max_compare(nlp, cherche(nlp, candidat_garde))

    # TODO: modify initial point here!!!(tiktak)

    x1 = best_x[ind_x1]
    x2 = best_x[ind_x2]
    x0[ind_x1] .= x1
    x0[ind_x2] .= x2

    # Mettre à jour dans le solveur / NLP
    #solver.x .= x0_finalMP.set_start_value(v, val)

    results = MadNLP.solve!(solver)

    # Get complementarity solution
    x1_sol = results.solution[ind_x1]
    x2_sol = results.solution[ind_x2]

    return results
end

model = load_model()
mpcc = convert_jump_to_mpcc(model)
results = tiktak(mpcc)
