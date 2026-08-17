
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


function push_inside(x, lb, ub; rel_margin = 1e-6, abs_margin = 1e-8)
    for i in eachindex(x)
        width = ub[i] - lb[i]

        if width <= 0.0
            x[i] = lb[i]
        else
            δ = min(width / 3, max(abs_margin, rel_margin * width))
            x[i] = clamp(x[i], lb[i] + δ, ub[i] - δ)
        end
    end

    return x
end



function score_candidat(nlp, x0, x_cc_candidat, ind_cc)
    
    x_full = copy(x0)
    x_full[ind_cc] .= x_cc_candidat


    # eval fonction objectif
    obj = try
        f = NLPModels.obj(nlp, x_full)
        isfinite(f) ? Float64(f) : Inf
    catch
        Inf
    end

    # Évaluation de la violation des contraintes (Faisabilité primale)
    violation = try
        c = NLPModels.cons(nlp, x_full)
        lcon = nlp.meta.lcon
        ucon = nlp.meta.ucon

        viol_sum = 0.0
        for i in eachindex(c)
            v_low = max(0.0, lcon[i] - c[i])
            v_upp = max(0.0, c[i] - ucon[i])
            viol_sum += v_low^2 + v_upp^2
        end

        sqrt(viol_sum)
    catch
        Inf
    end

    return violation, obj
end


function short_madnlp_polish(nlp::AbstractNLPModel, x0::Vector{Float64}, ind_cc;
                            max_iter,  
                            tol = 1e-6)
    try
        res = MadNLP.madnlp(
            nlp;
            x0 = x0,
            max_iter = max_iter,
            tol = tol,
            print_level = MadNLP.OFF
        )

        # Si aucune solution n'est retournée ou si elle contient des valeurs invalides
        if isnothing(res) || isnothing(res.solution) || any(!isfinite, res.solution)
            return copy(x0)
        end

        y = Vector{Float64}(res.solution)
        
        # S'assurer que la solution reste strictement dans les bornes
        y[ind_cc] .= push_inside(y[ind_cc], lb, ub)
        return y[ind_cc]

    catch
        # En cas de crash du solveur, on retourne x0 intact
        return x0[ind_cc]
    end
end


function tiktak(mpcc)
    
    solver = build_madnlp(mpcc)
    nlp = solver.cb.nlp
    # Get initial primal position
    x0 = NLPModels.get_x0(nlp)
    x1 = copy(x0)
    
    # x00 = copy(x0)
    

    # Get indices
    ind_x1 = mpcc.meta.ind_cc1
    ind_x2 = mpcc.meta.ind_cc2

    # TODO: modify initial point here!!!(tiktak)

    candidat = Vector{Vector{Float64}}()

    lvar = NLPModels.get_lvar(nlp)  
    uvar = NLPModels.get_uvar(nlp)  

    ind_cc = vcat(ind_x1, ind_x2)

    

    lb = lvar[ind_cc]
    ub = uvar[ind_cc]

    println("lb ", lb)
    println("ub ", ub)



    x0[ind_cc] .= push_inside(x0[ind_cc], lb, ub)
    println("point de deparrt ", x0)


    new_x0 = vcat(x0[ind_x1], x0[ind_x2])
    new_x0 = push_inside(new_x0, lb, ub)
    push!(candidat, new_x0)

    # echantillonage 

    # peut augmenter le nombre de tirage n_sample = 100 ici

    Xraw = QuasiMonteCarlo.sample(100, lb, ub, SobolSample())
    X = ndims(Xraw) == 1 ? reshape(collect(Xraw), length(lb), :) : Xraw
    for k in 1:size(X, 2)
        x = Vector{Float64}(X[:, k])
        x = push_inside(x, lb, ub)
        push!(candidat, x)
    end

    # Phase TIC 

    scores = [
    score_candidat(nlp, x0, x_cc, ind_cc) 
    for x_cc in candidat
    ]

    #println("score", scores)
    order = sortperm(eachindex(candidat), by = i -> scores[i])
    best_x = copy(candidat[order[1]])
    best_score = scores[order[1]]

    # Phase TAC

    # nombre des candidats à ameliorer 
    kmax = 15

    for r in 1:kmax
        x_sample = candidat[order[r]]

        # Mélange TikTac : combinaison entre le meilleur courant et un candidat.
        if r == 1
            y_samp = copy(x_sample)
        else
            λ = 0.25 + 0.75 * (r - 1) / max(1, kmax - 1)
            y_samp = (1 - λ) .* best_x .+ λ .* x_sample
        end

        y_samp = push_inside(y_samp, lb, ub)
        y = copy(x0)
        y[ind_cc] .= y_samp # on remet les autres valeurs du x0, rapel: x00 est une copie original du x0 de base
    
        best_x = short_madnlp_polish(nlp, y, ind_cc, max_iter = 40, tol = 1e-6)

    end

    best_x0 = copy(x0)
    best_x0[ind_cc] .= best_x 
    # Recompute initial position
    x1 = best_x0[ind_x1]
    x2 = best_x0[ind_x2]
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
