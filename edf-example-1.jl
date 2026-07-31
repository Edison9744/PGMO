using JuMP, Gurobi
using MathOptComplements
using Ipopt
using DelimitedFiles
using SCIP
using CSV
using DataFrames
using CCOpt
using NLPModelsJuMP

include("examples/edf-example-1/quad_jump.jl")
include("examples/edf-example-1/hs15jump.jl")

results = Matrix{Any}(zeros(82, 10))
k = 1


#A changer pour réutiliation du code
dossier_fichdat = "Documents/PGMO/pgmo-iroe-epec-main/pgmo-iroe-epec-main/scripts/examples/edf-example-1/optimTarif/run_tests/W10SGeq50"
for (root, dirs, files) in walkdir(dossier_fichdat)
    for file in files
        if startswith(file, "quad_")
            #println("Dossier actuel (root) : ", root)
            #chemin_fichier = joinpath(root,file)
            #println("fichier actuel : ", chemin_fichier)


            println("Etapes ", k)

            model = quad_model(joinpath(@__DIR__, "examples/edf-example-1/optimTarif/run_tests/W10SGeq50/$file"),
            )
            # Wrap in MathOptComplements
            #MathOptComplements.Bridges.add_all_bridges(model)
            # Define solver
            #Définir le solver directement
            set_optimizer(model, () -> MathOptComplements.Optimizer(Gurobi.Optimizer()))    
            set_attribute(model, "TimeLimit", 60.0)
            # Solve
            JuMP.set_silent(model)
            optimize!(model)

            status = JuMP.termination_status(model)
            c = Int(status)

            results[k,1] = "$file"
            #results[k,2] = JuMP.objective_value(model)
            #results[k,3] = c
            #results[k,4] = JuMP.solve_time(model)

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

writedlm("new_instance.csv", results, ';')

df = CSV.read("new_instance.csv", DataFrame)
println(df)


