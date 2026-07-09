using JuMP
using NLPModels
using NLPModelsJuMP
using Ipopt
using LinearAlgebra

model = Model(Ipopt.Optimizer)
@variable(model, x1 <= 0.5)
@variable(model, x2)
@objective(model, Min, 100.0 * (x2 - x1^2)^2 + (1.0 - x1)^2)
@constraint(model, x1 * x2 >= 1.0)
@constraint(model, x1 + x2^2 >= 0.0)

JuMP.optimize!(model)


# Get Ipopt object
optimizer = JuMP.unsafe_backend(model)

# Step 1: Recover primal-dual solution
# Primal solution as vector
x = optimizer.inner.x
# Dual solutions as vectors
y = optimizer.inner.mult_g
zl = optimizer.inner.mult_x_L
zu = optimizer.inner.mult_x_U


# Convert model as NLPModelsJuMP
nlp = MathOptNLPModel(model)

# Evaluate objective at solution
obj = NLPModels.obj(nlp, x)

grad_obj= NLPModels.grad(nlp,x)

#aucune contrainte linéaire
#Jx_1 = NLPModels.jac_lin(nlp, x)

Jx = NLPModels.jac_nln(nlp, x)

# dual dans l'ordre i.e dual = [multiplicateur d'égalité, multiplicateur d'inegalité]
# tout est évalué en un point x ici 
function verif_KKT(grad_obj, grad_cons, dual, ineq_cons=Float64[], eq_const=Float64[], tol=1e-5)

    somm = copy(grad_obj)

    for i in 1:length(dual)
        somm .+= dual[i] .* grad_cons[i]
    end

    statio = norm(somm) <= tol

    cond_KKT = Bool[]
    nb_eq = length(eq_cons)
    nb_ineq = length(ineq_cons)

    # Egalité contrainte nulle 
    for i in 1:nb_eq
        push!(cond_KKT, abs(eq_cons[i]) <= tol)
    end
    

    for j in 1:nb_ineq

        # Indice dans le vecteur dual global
        idx_dual = nb_eq + j 
        lambda = dual[idx_dual]
        valeur_contrainte = ineq_cons[j]

        # Admissibilité et signe du dual (λ >= 0 pour une contrainte <= 0), contrainte inactive à voir ? Ne fonctionne pas  
        #push!(cond_KKT, valeur_contrainte <= tol)
        #push!(cond_KKT, lambda >= -tol)
            
        # Si le dual est significativement positif, la contrainte doit être à 0
        if lambda > tol
            push!(cond_KKT, abs(valeur_contrainte) <= tol)
        end
    end 
        
        

    if all(cond_KKT) && statio 
        println("Toutes les conditions sont validées !")
    else
        println("les conditions ne sont pas validées !" )
    end
end 

    
ineq_cons = NLPModels.cons_nln(nlp,x)
dual = y

i, j = NLPModels.jac_structure(nlp)
v = NLPModels.jac_coord(nlp, x)
jac = NLPModels.sparse(i, j, v, 2, 2)

grad_cons = [jac[i, :] for i in 1:2]
println("jac", grad_cons[1])
eq_cons =[]

verif_KKT(grad_obj, grad_cons, dual, ineq_cons)





# TODO:
# [ ]  Evaluate gradient
# [ ]  Evaluate Jacobian of the constraints
# [ ]  Verify KKT equations
