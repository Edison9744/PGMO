using JuMP, Gurobi
using MathOptComplements
using Ipopt
using PATHSolver
using DelimitedFiles
using CSV
using DataFrames
using NLPModelsJuMP
using CCOpt

include("examples/edf-example-2/1d.jl")
include("new_start.jl")



# alphaa
list_model_a = Any[]
d=1
val_a = Matrix{Any}(zeros(100, 1))
for i in 0.1:0.1:10
    a= fill(i,10)
    model = edf_tarification_1d_model(alppha = a)
    push!(list_model_a, model)
    val_a[d,1]=i
    d+=1
end
writedlm("list_alpha.csv", val_a, ';')
println(length(list_model_a))



#betaa
list_model_b = Any[]
d=1
val_b = Matrix{Any}(zeros(100, 1))
for j in 0.1:0.1:10
    b= fill(j,10)
    model = edf_tarification_1d_model(betaa = b)
    push!(list_model_b, model)
    val_b[d,1] = j
    d+=1
end
writedlm("list_beta.csv", val_b, ';')
println(length(list_model_b))


function fichier(list_model, val)

    results = Matrix{Any}(zeros(length(list_model), 8))
    k = 1

    for model in list_model

        set_optimizer(model, () -> MathOptComplements.Optimizer(Ipopt.Optimizer()))    
        # Solve
        JuMP.set_silent(model)
        optimize!(model)

        
        c= Int(termination_status(model))

        results[k,1] = val[k,1]
        results[k,2] = JuMP.objective_value(model)
        results[k,3] = c
        results[k,4] = JuMP.solve_time(model)
            
        # au tour de Gurobi 

        println("Etape ", k," sur ", size(val, 1))
        set_optimizer(model, () -> MathOptComplements.Optimizer(Gurobi.Optimizer()))
        set_attribute(model, "TimeLimit", 300.0)

        #val_var = start_Ipopt(model; n_iter=10)

        for k in 1:size(val_var, 1) set_start_value(val_var[k, 1], val_var[k, 2]) end

        # Solve
        JuMP.set_silent(model)
        optimize!(model)


        c= Int(termination_status(model))
        results[k,5] = c

        if !JuMP.is_solved_and_feasible(model)
            results[k,6] = 0.0
        else
            results[k,6] = JuMP.objective_value(model)
        end

        results[k,7] = JuMP.solve_time(model)

        k+=1
    end 

    writedlm("benchmark_1d.csv", results, ';')
    df = CSV.read("benchmark_1d.csv", DataFrame; delim =';')
    return df
end

fichier(list_model_a, val_a)
fichier(list_model_b, val_b)