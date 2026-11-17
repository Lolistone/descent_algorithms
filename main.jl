using Plots

include("optimizer.jl")
include("benchmarks.jl")

# Problemas de prueba ------------------
# x0 es el punto inicial; x0_gd es el del descenso con paso fijo
# (con paso constante algunos puntos iniciales divergen).
struct Problem
    name::String
    f::Function
    g::Function
    h::Function
    x0::Vector{Float64}
    x0_gd::Vector{Float64}
    max_iter::Int
end

problems = [
    Problem("Sphere",     sphere,     sphere_g,     sphere_h,     [1.0, 2.0],              [2.0, 2.0],           100000),
    Problem("Rosenbrock", rosenbrock, rosenbrock_g, rosenbrock_h, [-1.0, 1.0],             [2.0, 2.0],           100000),
    Problem("Rastrigin",  rastrigin,  rastrigin_g,  rastrigin_h,  [-1.0, 1.0],             [2.0, 2.0],           10000),
    Problem("Wood",       wood,       wood_g,       wood_h,       [-3.0, -1.0, -3.0, -1.0], [2.0, 1.0, 1.0, 1.0], 10000),
    Problem("Froth",      froth,      froth_g,      froth_h,      [0.5, -2.0],             [0.5, -2.0],          1000),
]

# Métodos a comparar ------------------
methods = [
    ("Descenso paso constante",     GradientDescent(0.01)),
    ("Armijo",                      Armijo(1, 0.2, 2.0)),
    ("Newton modificado + Armijo",  Newton(1, 0.2, 2.0, 0.1)),
    ("Gradientes conjugados",       GradientesConjugados(1, 0.2, 2.0)),
]

# Matriz de desempeño (iteraciones; FALLO si no converge).
const FALLO = 1e8
C = zeros(length(methods), length(problems))

for (s, (label, opt)) in enumerate(methods)
    println("=== $label ===")
    for (p, prob) in enumerate(problems)
        println("$(prob.name): ")
        x_0 = opt isa GradientDescent ? prob.x0_gd : prob.x0
        res = descent(opt, prob.f, prob.g, prob.h, x_0, prob.max_iter)
        println("----------------------------------")

        # max(iter, 1) evita dividir por cero si x_0 ya es óptimo.
        C[s, p] = res.converge ? max(res.iter, 1) : FALLO
    end
end

# Tasa de desempeño relativa.
# Si el método falla, la tasa es infinita.
r = C ./ minimum(C, dims = 1)
r[C .== FALLO] .= Inf

# Desempeño del algoritmo s, y tasa de desempeño relativa t.
function desempeño(r, s, t)
    fila = r[s, :]
    return count(x -> x <= t, fila) / length(fila)
end

# Plot de desempeño
t_vals = 1:250

gr()
p = plot(xlabel = "t", ylabel = "ρ(t)", legend = :bottomright)

for (s, (label, _)) in enumerate(methods)
    y_vals = [desempeño(r, s, t) for t in t_vals]
    plot!(p, t_vals, y_vals, label = label)
end

# Se guarda en img/ para referenciarla desde el README.
mkpath("img")
savefig(p, "img/desempeno.png")
