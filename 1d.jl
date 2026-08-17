using JuMP
using MathOptComplements
using Ipopt

include("data_1d_10.jl")

# 1. Setup Model
function edf_tarification_1d_model(; alppha= fill(1.0, 10), betaa = fill(0.01, 10))
    model = Model()

    # -----------------------------------------------------------------
    # 2. Sets and Data (Placeholders - replace with your actual data)
    # -----------------------------------------------------------------
    T = 1:24            # Time steps
    K = 1:10            # Clients
    N = 1:2             # Tariff periods

    # Parameters
    epsilon = 1e-6
    pL = 0.05
    pU = 0.35

    ALPHA = alppha
    BETA = betaa
    FLEX = fill(0.1, length(K))

    # -----------------------------------------------------------------
    # 3. Variables
    # -----------------------------------------------------------------

    # Dual Variables
    @variable(model, lambda_x[T, K] >= 0)
    @variable(model, mu_x[T, K] >= 0)
    @variable(model, gamma_x[K])
    @variable(model, lambda_r[T, K] >= 0)
    @variable(model, mu_r[T, K] >= 0)
    @variable(model, gamma_r[K])
    @variable(model, lambda_y[K] >= 0)
    @variable(model, mu_y[K] >= 0)

    # Primal Variables
    @variable(model, pL <= q[N] <= pU)
    @variable(model, 0 <= h[T, N] <= 1)
    @variable(model, eps_var[T, N] >= 0) # 'eps' is a keyword in Julia, using eps_var
    @variable(model, pL <= price[T] <= pU)
    @variable(model, 0 <= y[K] <= 1)
    @variable(model, x[T, K])
    @variable(model, r[T, K])
    @variable(model, BILL_REF[K])

    # -----------------------------------------------------------------
    # 4. Constraints & Complementarity
    # -----------------------------------------------------------------

    # Stationarity Constraints
    @constraint(model, stationarity_r[t in T, k in K],
        OUTSIDE_PER_KWH[t] + (r[t, k] - CONSO_REF[t, k]) / ALPHA[k] +
        lambda_r[t, k] - mu_r[t, k] - gamma_r[k] == 0)

    @constraint(model, stationarity_x[t in T, k in K],
        price[t] + (x[t, k] - CONSO_REF[t, k]) / ALPHA[k] +
        lambda_x[t, k] - mu_x[t, k] - gamma_x[k] == 0)

    @constraint(model, stationarity_y[k in K],
        sum(price[t] * 365.25 * CONSO_REF[t, k] for t in T) - BILL_REF[k] +
        (1 / BETA[k]) * (y[k] - 0.5) + lambda_y[k] - mu_y[k] == 0)

    # Complementarity Constraints using MOI.Complements
    @constraint(model, complementarity_r_lb[t in T, k in K],
        [x[t, k] - (1 - FLEX[k]) * CONSO_REF[t, k], mu_r[t, k]] in MOI.Complements(2))

    @constraint(model, complementarity_r_ub[t in T, k in K],
        [(1 + FLEX[k]) * CONSO_REF[t, k] - x[t, k], lambda_r[t, k]] in MOI.Complements(2))

    @constraint(model, complementarity_x_lb[t in T, k in K],
        [x[t, k] - (1 - FLEX[k]) * CONSO_REF[t, k], mu_x[t, k]] in MOI.Complements(2))

    @constraint(model, complementarity_x_ub[t in T, k in K],
        [ (1 + FLEX[k]) * CONSO_REF[t, k] - x[t, k], lambda_x[t, k]] in MOI.Complements(2))

    @constraint(model, complementarity_y_ub[k in K], [1 - y[k], lambda_y[k]] in MOI.Complements(2))

    @constraint(model, complementarity_y_lb[k in K], [y[k], mu_y[k]] in MOI.Complements(2))

    # Primal Bounds and Physical Constraints
    @constraint(model, r_bounds_lb[t in T, k in K], r[t, k] >= (1 - FLEX[k]) * CONSO_REF[t, k])
    @constraint(model, r_bounds_ub[t in T, k in K], r[t, k] <= (1 + FLEX[k]) * CONSO_REF[t, k])
    @constraint(model, r_marginal[k in K], sum(r[t, k] for t in T) == sum(CONSO_REF[t, k] for t in T))

    @constraint(model, x_bounds_lb[t in T, k in K], x[t, k] >= (1 - FLEX[k]) * CONSO_REF[t, k])
    @constraint(model, x_bounds_ub[t in T, k in K], x[t, k] <= (1 + FLEX[k]) * CONSO_REF[t, k])
    @constraint(model, x_marginal[k in K], sum(x[t, k] for t in T) == sum(CONSO_REF[t, k] for t in T))

    @constraint(model, outside_bill_def[k in K], BILL_REF[k] == sum(OUTSIDE_PER_KWH[t] * 365.25 * r[t, k] for t in T))

    # McCormick / Tariff Logic
    @constraint(model, mccormick_lb[t in T, n in N], q[n] - (1 - h[t, n]) * pU <= price[t])
    @constraint(model, mccormick_ub[t in T, n in N], price[t] <= q[n] + (1 - h[t, n]) * pU)

    @constraint(model, ramp_ub[n in N, t in T[2:end]], h[t, n] - h[t-1, n] <= eps_var[t, n])
    @constraint(model, ramp_lb[n in N, t in T[2:end]], h[t-1, n] - h[t, n] <= eps_var[t, n])

    @constraint(model, prices_ordering[n in 1:(length(N)-1)], q[n] <= q[n+1])
    @constraint(model, allocation[t in T], sum(h[t, n] for n in N) == 1)

    @constraint(model, switch_limit_1, sum(eps_var[t, 1] for t in T) <= 4)
    @constraint(model, switch_limit_2, sum(eps_var[t, 2] for t in T) <= 4)
    @constraint(model, min_length_1, sum(h[t, 1] for t in T) == 8)
    @constraint(model, min_length_2, sum(h[t, 2] for t in T) == 16)

    # -----------------------------------------------------------------
    # 5. Objective
    # -----------------------------------------------------------------
    # Note: complementarity solvers often solve the feasibility problem (KKT system).
    # If you are solving this as an Optimization problem with MPEC constraints,
    # you should use a solver like Knitro or Ipopt.

    @objective(model, Max, sum(K_CLIENTS_WEIGHT[k] * (sum((price[t] - COST_PER_KWH[t]) * 365.25 * x[t, k] * y[k] for t in T))
       for k in K))

    # objective reformulé 2
    #@objective(model, Max, sum( sum( -y[k]*sum( (x[t,k]-CONSO_REF[t,k])^2 ) /(2*ALPHA[k]) - y[k]^2/BETA[k] - lambda_y[k] + (BILL_REF[k] - COST_PER_KWH[t] * 365.25 * x[t, k] + 1/(2*BETA[k]))* y[k]   for t in T)  for k in K))
    return model
end

