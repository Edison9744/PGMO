using JuMP
using NLPModels
using NLPModelsJuMP
using Ipopt
using LinearAlgebra

#model test

model = Model(Ipopt.Optimizer)
@variable(model, x1 <= 0.5)
@variable(model, x2)
@objective(model, Min, 100.0 * (x2 - x1^2)^2 + (1.0 - x1)^2)
@constraint(model, x1 * x2 >= 1.0)
@constraint(model, x1 + x2^2 >= 0.0)
JuMP.set_attribute(model,"tol",1e-4 )
JuMP.optimize!(model)


function verif_KKT_ipopt(model)

    optimizer = JuMP.unsafe_backend(model)
    nlp = MathOptNLPModel(model)   

    
    l = NLPModels.get_lcon(nlp)
    u = NLPModels.get_ucon(nlp)

    xl = NLPModels.get_lvar(nlp)
    xu = NLPModels.get_uvar(nlp)

    # Donne les indices des inégalités et égalités de contrainte
    ind_eq = findall(l .== u)
    ind_ineq = findall(l .< u)

    
    x = optimizer.inner.x
    c = NLPModels.cons(nlp,x)

    # Dual solutions as vectors
    y = optimizer.inner.mult_g
    zl = optimizer.inner.mult_x_L
    zu = optimizer.inner.mult_x_U

    grad_obj= NLPModels.grad(nlp,x)
    jac = NLPModels.jac(nlp,x)


    # faisabilité duale 
    inf_du = norm(grad_obj .+ jac' * y .- zl .+ zu, Inf)

    #faisabilité primal
    nf_pr = norm(c .- clamp.(c, l, u), Inf)

    # contrainte de complémentarité

    inf_cc1 = norm(min.(x .- xl, zl), Inf)
    inf_cc2 = norm(min.(xu .- x, zu), Inf)

    yl = max.(-y[ind_ineq], Inf)
    yu = min.(y[ind_ineq], Inf)

    inf_cc3 = norm(min.(c[ind_ineq] .- l[ind_ineq], yl[ind_ineq]))
    inf_cc4 = norm(min.(u[ind_ineq] .- c[ind_ineq], yu[ind_ineq]))
   
    
    nf_compl = maximum([inf_cc1, inf_cc2, inf_cc3, inf_cc4])

    return inf_du, nf_pr, inf_cc1, inf_cc2, inf_cc3, inf_cc4, nf_compl
end 


verif_KKT_ipopt(model)






