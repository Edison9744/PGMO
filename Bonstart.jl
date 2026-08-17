using JuMP, Gurobi
using Ipopt
using MathOptInterface
using Random
using QuasiMonteCarlo # Pour l'échantillonnage de type Sobol/LHS
using LBFGSB          # Ou un optimiseur léger pour la phase TAC




include("examples/edf-example-2/1d.jl")

#const MOI = MathOptInterface


# struct NLPExtractor
#     evaluator::MOI.AbstractNLPEvaluator
#     num_vars::Int
#     g_lb::Vector{Float64}
#     g_ub::Vector{Float64}
# end

# function extract_nlp(model::Model)
#     # 1. Créer l'évaluateur JuMP
#     evaluator = JuMP.NLPEvaluator(model)
    
#     # 2. Initialiser l'évaluateur (OBLIGATOIRE avant toute requête)
#     MOI.initialize(evaluator, Symbol[:ExprGraph])
    
#     # 3. Extraire les bornes des contraintes non-linéaires
#     nlp_bounds = MOI.constraint_bounds(evaluator)
#     g_lb = [b.lower for b in nlp_bounds]
#     g_ub = [b.upper for b in nlp_bounds]
    
#     num_vars = num_variables(model)
#     return NLPExtractor(evaluator, num_vars, g_lb, g_ub)
# end

# # évaluer l'objectif f
# function eval_f(nlp::NLPExtractor, x::Vector{Float64})
#     return MOI.eval_objective(nlp.evaluator, x)
# end

# # évaluer la violation des contraintes V en un point x
# function eval_violation(nlp::NLPExtractor, x::Vector{Float64})
#     g = zeros(length(nlp.g_lb))
#     MOI.eval_constraint(nlp.evaluator, g, x)
    
#     violation = 0.0
#     for i in 1:length(g)
#         if g[i] < nlp.g_lb[i]
#             violation += (nlp.g_lb[i] - g[i])^2
#         elseif g[i] > nlp.g_ub[i]
#             violation += (g[i] - nlp.g_ub[i])^2
#         end
#     end
#     return violation
# end


# function tictac_start_point(nlp, lb::Vector{Float64}, ub::Vector{Float64};
#                             N_samples=100, K_top=10, interior_push_eps=1e-3)
    
#     dim = nlp.num_vars
    
#     # --- PHASE TIC : Échantillonnage Sobol (Quasi-Monte Carlo)
#     samples = QuasiMonteCarlo.sample(N_samples, lb, ub, SobolSample())
    
#     scores = Float64[]
#     candidates = Vector{Float64}[]
    
#     for i in 1:N_samples
#         x_cand = samples[:, i]
#         f_val = eval_f(nlp, x_cand)
#         v_val = eval_violation(nlp, x_cand)
        
#         # Le score privilégie fortement la réalisation des contraintes
#         score = 1e3 * v_val + f_val
        
#         push!(candidates, x_cand)
#         push!(scores, score)
#     end
    
#     # Sélection des K meilleurs candidats
#     top_indices = sortperm(scores)[1:min(K_top, N_samples)]
#     best_candidates = candidates[top_indices]
    
#     # --- PHASE TAC : Projection locale rapide & Interior Push ---
#     best_x = copy(best_candidates[1])
#     best_v = Inf
#     best_f = Inf
    
#     for x_start in best_candidates
#         # On applique un ajustement local rapide (ici une recherche locale simplifiée)
#         # Pour projeter le point vers la zone réalisable
#         x_projected = copy(x_start)
        
#         # Pousser les points à l'intérieur des bornes (Interior Push pour Ipopt)
#         for j in 1:dim
#             x_projected[j] = clamp(x_projected[j], lb[j] + interior_push_eps, ub[j] - interior_push_eps)
#         end
        
#         v_val = eval_violation(nlp, x_projected)
#         f_val = eval_f(nlp, x_projected)
        
#         # Choisir la meilleure solution trouvée
#         if v_val < best_v || (isapprox(v_val, best_v, atol=1e-4) && f_val < best_f)
#             best_v = v_val
#             best_f = f_val
#             best_x = copy(x_projected)
#         end
#     end
    
#     return best_x
# end








const MOI = MathOptInterface


function _box_from_model(model::JuMP.Model; fallback_bound = 10.0)
    vars = JuMP.all_variables(model)

    lb = zeros(length(vars))
    ub = zeros(length(vars))

    for (i, v) in enumerate(vars)
        if JuMP.is_fixed(v)
            val = Float64(JuMP.fix_value(v))
            lb[i] = val
            ub[i] = val
            continue
        end

        l = JuMP.has_lower_bound(v) ? Float64(JuMP.lower_bound(v)) : -Inf
        u = JuMP.has_upper_bound(v) ? Float64(JuMP.upper_bound(v)) : Inf

        lb[i] = l
        ub[i] = u
    end

    return vars, lb, ub
end

function _push_inside!(x, lb, ub; rel_margin = 1e-6, abs_margin = 1e-8)
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

function _point_dict(vars, x)
    return Dict(vars[i] => x[i] for i in eachindex(vars))
end

function _objective_score(model::JuMP.Model, point::Dict)
    sense = JuMP.objective_sense(model)

    if sense == MOI.FEASIBILITY_SENSE
        return 0.0
    end

    val = try
        JuMP.value(v -> point[v], JuMP.objective_function(model))
    catch
        return Inf
    end

    if !isfinite(val)
        return Inf
    end

    # On transforme toujours en problème de minimisation pour le score.
    return sense == MOI.MAX_SENSE ? -Float64(val) : Float64(val)
end

function _score(model::JuMP.Model, vars, x)
    point = _point_dict(vars, x)

    violation = try
        report = JuMP.primal_feasibility_report(model, point; atol = 0.0)

        s = 0.0
        for v in values(report)
            s += abs2(Float64(v))
        end

        sqrt(s)
    catch
        Inf
    end

    obj = _objective_score(model, point)

    return violation, obj
end

function _better(score_a, score_b; viol_tol = 1e-8)
    viol_a, obj_a = score_a
    viol_b, obj_b = score_b

    return viol_a < viol_b - viol_tol ||
           abs(viol_a - viol_b) <= viol_tol && obj_a < obj_b
end

function _short_ipopt_polish(model::JuMP.Model, vars, x0, lb, ub;
                             max_iter = 40,
                             tol = 1e-6)

    # On travaille sur une copie pour ne pas modifier le modèle original.
    local m, ref
    try
        m, ref = JuMP.copy_model(model)
    catch
        return copy(x0)
    end

    copied_vars = [ref[v] for v in vars]

    for (v, val) in zip(copied_vars, x0)
        JuMP.set_start_value(v, val)
    end

    JuMP.set_optimizer(m, Ipopt.Optimizer)
    JuMP.set_silent(m)

    JuMP.set_optimizer_attribute(m, "print_level", 0)
    JuMP.set_optimizer_attribute(m, "max_iter", max_iter)
    JuMP.set_optimizer_attribute(m, "tol", tol)

    try
        JuMP.optimize!(m)
    catch
        return copy(x0)
    end

    if !JuMP.has_values(m)
        return copy(x0)
    end

    y = [Float64(JuMP.value(v)) for v in copied_vars]

    if any(!isfinite, y)
        return copy(x0)
    end

    return _push_inside!(y, lb, ub)
end

# ------------------------------------------------------------
# Fonction principale TikTac
# ------------------------------------------------------------

function tictac_start!(model::JuMP.Model;
                       n_samples = 100,
                       n_polish = 10,
                       polish_iter = 40,
                       fallback_bound = 10.0,
                       verbose = true)

    vars, lb, ub = _box_from_model(model; fallback_bound = fallback_bound)

    if isempty(vars)
        return Dict{JuMP.VariableRef, Float64}()
    end

    candidates = Vector{Vector{Float64}}()

    # 1) Point milieu
    mid = (lb .+ ub) ./ 2
    _push_inside!(mid, lb, ub)
    push!(candidates, copy(mid))

    # 2) Point de départ déjà présent dans le modèle, s'il existe
    starts = JuMP.start_value.(vars)
    x_start = [
        isnothing(starts[i]) ? mid[i] : Float64(starts[i])
        for i in eachindex(vars)
    ]
    _push_inside!(x_start, lb, ub)
    push!(candidates, x_start)

    # 3) Échantillonnage Sobol
    if n_samples > 0
        Xraw = QuasiMonteCarlo.sample(n_samples, lb, ub, SobolSample())
        X = ndims(Xraw) == 1 ? reshape(collect(Xraw), length(lb), :) : Xraw

        for k in 1:size(X, 2)
            x = Vector{Float64}(X[:, k])
            _push_inside!(x, lb, ub)
            push!(candidates, x)
        end
    end

    # Phase TIC : on score tous les candidats
    scores = [_score(model, vars, x) for x in candidates]
    order = sortperm(eachindex(candidates), by = i -> scores[i])

    best_x = copy(candidates[order[1]])
    best_score = scores[order[1]]

    # Phase TAC : on améliore localement quelques meilleurs candidats
    kmax = min(n_polish, length(order))

    for r in 1:kmax
        x_sample = candidates[order[r]]

        # Mélange TikTac : combinaison entre le meilleur courant et un candidat.
        if r == 1
            x0 = copy(x_sample)
        else
            λ = 0.25 + 0.75 * (r - 1) / max(1, kmax - 1)
            x0 = (1 - λ) .* best_x .+ λ .* x_sample
        end

        _push_inside!(x0, lb, ub)

        y = _short_ipopt_polish(
            model,
            vars,
            x0,
            lb,
            ub;
            max_iter = polish_iter,
        )

        y_score = _score(model, vars, y)

        if _better(y_score, best_score)
            best_x = copy(y)
            best_score = y_score
        end
    end

    # On écrit le meilleur point trouvé comme point initial du modèle original.
    for (v, val) in zip(vars, best_x)
        JuMP.set_start_value(v, val)
    end

    if verbose
        println("TikTac terminé.")
        println("Violation finale approx. : ", best_score[1])
        println("Score objectif approx.   : ", best_score[2])
    end

    return Dict(vars[i] => best_x[i] for i in eachindex(vars))
end





# a = fill(0.1,10)
# model = edf_tarification_1d_model(alppha = a)
function init_model_tik(model)
    set_optimizer(model, () -> MathOptComplements.Optimizer(Ipopt.Optimizer()))   
    #set_attribute(model, "TimeLimit", 120.0) 

    x0 = tictac_start!(
        model;
        n_samples = 200,
        n_polish = 12,
        polish_iter = 30,
    )

    vars = all_variables(model)

    # Vérification : le nombre de variables doit correspondre à la taille de x0
    if length(vars) == length(x0)
        for (var, val) in x0
            set_start_value(var, val)
        end
    else
        error("La taille de x0 ($(length(x0))) ne correspond pas au nombre de variables du modèle ($(length(vars))).")
    end

    JuMP.set_silent(model)
    
end

# optimize!(model)
      
# termination_status(model)
# JuMP.objective_value(model)
# JuMP.solve_time(model)
# JuMP.barrier_iterations(model)
