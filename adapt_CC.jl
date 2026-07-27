using JuMP
using NLPModels
using NLPModelsJuMP
using Ipopt
using SparseArrays
using LinearAlgebra

# ou p est la dimension de w0 ou w=[w0,w1,w2]; w1 et w2 de dim m
function indiq_direc(w,p,m)
    # les valeurs non connue vont prendre des 3 dans d
    d = J(p+2*m,1,3);
    w1 = [w[j] for j in p+1:m]
    w2 = [w[j] for j in m+1:2*m]

    for i in 1:m 
        # I+0
        if w1[i] > 0 && w2[i] == 0
            d[p+i] = 0
        end

        # I0+
        if w1[i] == 0 && w2[i] > 0
            d[p+m+i] = 0
        end
    end

    return d
end

function direction_cone(model,p,m)

    tol = 1e-8

    optimizer = JuMP.unsafe_backend(model)
    # Convert model as NLPModelsJuMP
    nlp = MathOptNLPModel(model)

    # Dual solutions as vectors
    # y = optimizer.inner.mult_g
    
    # Primal solution as vector
    w = optimizer.inner.x

    direc = indiq_direc(w,p,m) # les valeurs non connue contiennent des 3


    # De ceci il faut réussir à extraire les contrainte d'égalité et les contraintes d'inégalité
    # 1 : on récupère les indices d'inegalité et egalité, 2: on vérifie les indices d'inégalité actives 

    stats = ipopt(nlp)
    y = stats.multipliers 

    # Indices des contraintes d'égalité
    indices_egalites = nlp.meta.ieqn

    # Indices des contraintes d'inégalité
    # Cela inclut les contraintes bornées à gauche, à droite, ou des deux côtés
    indices_inegalites = [nlp.meta.ilow; nlp.meta.iupp; nlp.meta.irng]

    # Une inégalité est active si sont multiplicateur est strict positif
    inegalites_actives = [i for i in indices_inegalites if abs(y[i]) > tol ]

    # On va donc ensuite utiliser ces indices contraintes pour extraires les gradient des contraintes dans le jacobien

    ligne, col = jac_structure(nlp)
    v = jac_coord(nlp, w)

    # ou nlp.meta.ncon, nlp.meta.nvar donne la dim de la jacobienne : nombre de contrainte et nombre de variable
    J_sparse = sparse(ligne, col, v, nlp.meta.ncon, nlp.meta.nvar)

    gradients_actifs_ineq = J_sparse[inegalites_actives, :]
    gradients_eq   = J_sparse[indices_egalites, :]
    grad_obj= NLPModels.grad(nlp,w)


    # Début de l'optim sur d 
    
    model_d = Model(Ipopt.optimizer)

    
    @variable(model_d, d[1:p+2*m])
    @objective(model_d, Min, grad_obj*d)
    @constraint(model_d, [i in 1:(p+2*m); direc[i] == 0], d[i] == 0)

    for j in 1: length(gradients_actifs_ineq)
        @constraint(model_d, gradients_actifs_ineq[j]*d >= 0 )
    end
    

    for k in 1: length(gradients_eq)
        @constraint(model_d, gradients_eq[k]*d == 0 )
    end

    # @constraint(model_d, gradients_actifs_ineq*d .>= 0 )
    # @constraint(model_d, gradients_eq*d .== 0 )

    # Contrainte de complementarité
    
    w1 = [w[j] for j in p+1:m]
    w2 = [w[j] for j in m+1:2*m]

    for i in 1:m 
        # I00
        if w1[i] == 0 && w2[i] == 0
            @constraint(model_d, d[p+i] ⟂ d[p+m+i])
        end
    end
    optimize!(model_d)

    return JuMP.objective_value(model_d)
end