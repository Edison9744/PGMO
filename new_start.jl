using JuMP, Gurobi
using MathOptComplements
using NLPModelsJuMP
using Ipopt
using DelimitedFiles
using SCIP
using CSV
using DataFrames
using CCOpt





function start_Ipopt(model; n_iter=100)

    k=1
    vars = all_variables(model)
    val_var = Matrix{Any}(zeros(length(vars), 2))
    

    MathOptComplements.Bridges.add_all_bridges(model)
    #set_optimizer(model, CCOpt.Optimizer)
    
    set_optimizer(model, () -> MathOptComplements.Optimizer(Ipopt.Optimizer()))
    set_optimizer_attribute(model, "max_iter", n_iter)
    # Solve
    JuMP.set_silent(model)
    optimize!(model)

    # récupération
    for var in vars
        val_var[k,1] = var
        val_var[k,2] = value(var)
        k+=1
    end
    println("status ",termination_status(model) )
    println("solution trouvé par Ipopt ",objective_value(model))
    #println("nombre d'itération ", iteration_count(model))

    # obligé sinon le model reste avec un nombre d'itération max fixé avant 
    set_optimizer_attribute(model, "max_iter", 10000)
    return val_var
end



#= include("examples/edf-example-1/quad_jump.jl")

results = Matrix{Any}(zeros(85, 10))
k = 1

#A changer pour réutiliation du code
dossier_fichdat = "Documents/PGMO/pgmo-iroe-epec-main/pgmo-iroe-epec-main/scripts/examples/edf-example-1/optimTarif/run_tests/W10SGeq50"
for (root, dirs, files) in walkdir(dossier_fichdat)
    for file in files
        if startswith(file, "quad_")

            println("Etapes ", k)

            model = quad_model(joinpath(@__DIR__, "examples/edf-example-1/optimTarif/run_tests/W10SGeq50/$file"),
            )
            # Wrap in MathOptComplements
            #MathOptComplements.Bridges.add_all_bridges(model)
            # Define solver
            #Définir le solver directement
            # Define solver
            set_optimizer(model, () -> MathOptComplements.Optimizer(Gurobi.Optimizer()))
            set_attribute(model, "TimeLimit", 60.0)

            val_var = start_Ipopt(model)

            for k in 1:size(val_var, 1) set_start_value(val_var[k, 1], val_var[k, 2]) end

            # Solve
            JuMP.set_silent(model)
            optimize!(model)

            status = JuMP.termination_status(model)
            c = Int(status)

            results[k,1] = "$file"
            results[k,2] = JuMP.objective_value(model)
            results[k,3] = c
            results[k,4] = JuMP.solve_time(model)

            # au tour de CCopt

            MathOptComplements.Bridges.add_all_bridges(model)

            set_optimizer(model, CCOpt.Optimizer)
            JuMP.set_silent(model)
            optimize!(model)


            status = JuMP.termination_status(model)
            c = Int(status)

            results[k,5] = JuMP.objective_value(model)
            results[k,6] = c
            results[k,7] = JuMP.solve_time(model)


            # au tour de Ipopt 


            #Définir le solver directement
            set_optimizer(model, () -> MathOptComplements.Optimizer(Ipopt.Optimizer()))    
            # Solve
            JuMP.set_silent(model)
            optimize!(model)

            status = JuMP.termination_status(model)
            c = Int(status)

            results[k,8] = JuMP.objective_value(model)
            results[k,9] = c
            results[k,10] = JuMP.solve_time(model)
            
            k +=1
            
        end
    end
end

writedlm("start_instance_W10.csv", results, ';') =#



