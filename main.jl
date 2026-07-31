
using JuMP, Gurobi
using MathOptComplements
using Ipopt
using PATHSolver
using DelimitedFiles
using CSV
using DataFrames
using NLPModelsJuMP
using CCOpt


# include("examples/edf-example-2/1d.jl")
include("examples/edf-example-2/4d.jl")
model_2 = edf_tarification_1d_model()
model_3 = edf_tarification_4d_model()

# Wrap in MathOptComplements
# MathOptComplements.Bridges.add_all_bridges(model_2)


include("examples/macepec.jl")
include("examples/monteiro.jl") #NOPE
include("examples/nodal_zonal_arbitrage.jl")
include("examples/nodal_zonal_arbitrage_large.jl") #NOPE
include("examples/market_power.jl")


include("examples/risky_investment.jl")
PATHSolver.c_api_License_SetString("1259252040&Courtesy&&&USR&GEN2035&5_1_2026&1000&PATH&GEN&31_12_2035&0_0_0&6000&0_0")
model_1 = risky_investment(num_yrs=2, Δt=Int(8760/2), contract=:incomplete)
MathOptComplements.Bridges.add_all_bridges(model_1)
# Define solver
set_optimizer(model_1, CCOpt.Optimizer)
# set_optimizer(model, () -> PATHSolver.Optimizer())
JuMP.set_silent(model_1)
optimize!(model_1)
objective_value(model_1)



modd = Any[]
# push!(modd, market_power(H = 1))
push!(modd, model_1, model_2, model_3)
push!(modd, nodal_zonal_arbitrage_model())
push!(modd, electricity_market_model(), electric002_model(), electric004_model(), ex001_epec_model(), ex4_epec_model(), outrata3_epec_model(), outrata4_epec_model())

println("la liste de model est ", modd)


function fichier(list_model)

    results = Matrix{Any}(zeros(13, 8))
    k = 1

    for model in list_model

        set_optimizer(model, () -> MathOptComplements.Optimizer(Gurobi.Optimizer()))    
        set_attribute(model, "TimeLimit", 300.0)
        # Solve
        JuMP.set_silent(model)
        optimize!(model)

        if !JuMP.is_solved_and_feasible(model) || termination_status(model) == MOI.TIME_LIMIT
            c = 0
        else
            c = 1
        end

        println("--- Résultat du fichier file", k)
        results[k,1] = k

        if c==0
            results[k,2] = 0.0
        else
            results[k,2] = JuMP.objective_value(model)
        end

        results[k,3] = c
        results[k,4] = JuMP.solve_time(model)
                
        
        # au tour de Ipopt 

        set_optimizer(model, () -> MathOptComplements.Optimizer(Ipopt.Optimizer()))
        # Solve
        JuMP.set_silent(model)
        optimize!(model)

        if !JuMP.is_solved_and_feasible(model) || termination_status(model) == MOI.TIME_LIMIT
            c = 0
        else
            c = 1
        end

        results[k,5] = c

        if c==0
            results[k,6] = 0.0
        else
            results[k,6] = JuMP.objective_value(model)
        end

        results[k,7] = JuMP.solve_time(model)
        #results[k,8] = norm(y, Inf)

        #results[k,6] = norm(y, Inf)
        #_,results[k,7] = verif_KKT_ipopt(model)

        k +=1
    end
    writedlm("benchmark2_1.csv", results, ';')
    df = CSV.read("benchmark2_1.csv", DataFrame; delim =';')
    return df
end 
fichier(modd)












function fichier1(list_model)

    results = Matrix{Any}(zeros(13, 4))
    k = 1

    for model in list_model

        # Wrap in MathOptComplements
        MathOptComplements.Bridges.add_all_bridges(model)

        # Define solver
        set_optimizer(model, CCOpt.Optimizer)    
        # Solve
        JuMP.set_silent(model)
        optimize!(model)

        if !JuMP.is_solved_and_feasible(model) || termination_status(model) == MOI.TIME_LIMIT
            c = 0
        else
            c = 1
        end

        println("--- Résultat du fichier file", k)
        results[k,1] = k

        if c==0
            results[k,2] = 0.0
        else
            results[k,2] = JuMP.objective_value(model)
        end

        results[k,3] = c
        results[k,4] = JuMP.solve_time(model)
                

        k +=1
    end
    writedlm("benchmark_ex2.csv", results, ';')
    df = CSV.read("benchmark_ex2.csv", DataFrame; delim =';')
    return df
end 

fichier1(modd)










