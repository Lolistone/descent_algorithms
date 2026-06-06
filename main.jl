using Plots

include("optimizer.jl")
include("benchmarks.jl")

# Matriz de desempeño.
C = zeros(4, 5)

# Test descenso por gradiente con paso fijo.
grad_descent = GradientDescent(0.01)
x_0 = [2.0, 2.0]

# Sphere
println("Sphere:")
x, norma, converge, iter = descent(grad_descent, sphere_g, x_0, 10000)
println("----------------------------------")

if converge
    C[1, 1] = iter
else  
    C[1, 1] = 1e8
end

# Rosenbrock
println("Rosenbrock: ")
x, norma, converge, iter = descent(grad_descent, rosenbrock_g, x_0, 10000)
println("----------------------------------")

if converge
    C[1, 2] = iter
else  
    C[1, 2] = 1e8
end

# Rastrigin
println("Rastrigin: ")
x, norma, converge, iter = descent(grad_descent, rastrigin_g, x_0, 10000)
println("----------------------------------")

if converge
    C[1, 3] = iter
else  
    C[1, 3] = 1e8
end

# Wood
x_0 = [2.0, 1.0, 1.0, 1.0] 
println("Wood: ")
x, norma, converge, iter = descent(grad_descent, wood_g, x_0, 10000)
println("----------------------------------")

if converge
    C[1, 4] = iter
else  
    C[1, 4] = 1e8
end

# Froth
x_0 = [0.5, -2.0] 
println("Froth: ")
x, norma, converge, iter = descent(grad_descent, froth_g, x_0, 1000)
println("----------------------------------")

if converge
    C[1, 5] = iter
else  
    C[1, 5] = 1e8
end

# Test de descenso por gradiente con regla de armijo.
armijo = Armijo(1, 0.2, 2.0)

# Sphere
println("Sphere:")
x_0 = [1.0, 2.0]
x, norma, converge, iter = descent(armijo, sphere, x_0, sphere_g, 100000)
println("----------------------------------")

if converge
    C[2, 1] = iter
else  
    C[2, 1] = 1e8
end

# Rosenbrock
println("Rosenbrock: ")
x_0 = [-1.0, 1.0]
x, norma, converge, iter = descent(armijo, rosenbrock, x_0, rosenbrock_g, 10000)
println("----------------------------------")

if converge
    C[2, 2] = iter
else  
    C[2, 2] = 1e8
end

# Rastrigin
println("Rastrigin: ")
x, norma, converge, iter = descent(armijo, rastrigin, x_0, rastrigin_g, 10000)
println("----------------------------------")

if converge
    C[2, 3] = iter
else  
    C[2, 3] = 1e8
end

# Wood
x_0 = [-3.0, -1.0, -3.0, -1.0] 
println("Wood: ")
x, norma, converge, iter = descent(armijo, wood, x_0, wood_g, 1000)
println("----------------------------------")

if converge
    C[2, 4] = iter
else  
    C[2, 4] = 1e8
end

# Froth
x_0 = [0.5, -2.0] 
println("Froth: ")
x, norma, converge, iter = descent(armijo, froth, x_0, froth_g, 1000)
println("----------------------------------")

if converge
    C[2, 5] = iter
else  
    C[2, 5] = 1e8
end

# Test de método de Newton + regla de armijo.
newton = Newton(1, 0.2, 2.0, 0.1)

# Sphere
println("Sphere:")
x_0 = [1.0, 2.0]
x, norma, converge, iter = descent(newton, x_0, sphere, sphere_g, sphere_h, 100000)
println("----------------------------------")

if converge
    C[3, 1] = iter
else  
    C[3, 1] = 1e8
end

# Rosenbrock
println("Rosenbrock: ")
x_0 = [-1.0, 1.0]
x, norma, converge, iter = descent(newton, x_0, rosenbrock, rosenbrock_g, rosenbrock_h, 100000)
println("----------------------------------")

if converge
    C[3, 2] = iter
else  
    C[3, 2] = 1e8
end

# Rastrigin
println("Rastrigin: ")
x, norma, converge, iter = descent(newton, x_0, rastrigin, rastrigin_g, rastrigin_h, 10000)
println("----------------------------------")

if converge
    C[3, 3] = iter
else  
    C[3, 3] = 1e8
end

# Wood
x_0 = [-3.0, -1.0, -3.0, -1.0] 
println("Wood: ")
x, norma, converge, iter = descent(newton, x_0, wood, wood_g, wood_h, 1000)
println("----------------------------------")

if converge
    C[3, 4] = iter
else  
    C[3, 4] = 1e8
end

# Froth
x_0 = [0.5, -2.0] 
println("Froth: ")
x, norma, converge, iter = descent(newton, x_0, froth, froth_g, froth_h,1000)
println("----------------------------------")

if converge
    C[3, 5] = iter
else  
    C[3, 5] = 10e8
end

# Gradientes conjugados.

opt = GradientesConjugados(1, 0.2, 2.0)

# Sphere
println("Sphere:")
x_0 = [1.0, 2.0]
x, norma, converge, iter = descent(opt, x_0, sphere, sphere_g, 100000)
println("----------------------------------")

if converge
    C[4, 1] = iter
else  
    C[4, 1] = 1e8
end

# Rosenbrock
println("Rosenbrock: ")
x_0 = [-1.0, 1.0]
x, norma, converge, iter = descent(opt, x_0, rosenbrock, rosenbrock_g, 100000)
println("----------------------------------")

if converge
    C[4, 2] = iter
else  
    C[4, 2] = 1e8
end

# Rastrigin
println("Rastrigin: ")
x, norma, converge, iter = descent(opt, x_0, rastrigin, rastrigin_g, 10000)
println("----------------------------------")

if converge
    C[4, 3] = iter
else  
    C[4, 3] = 1e8
end

# Wood
x_0 = [-3.0, -1.0, -3.0, -1.0] 
println("Wood: ")
x, norma, converge, iter = descent(opt, x_0, wood, wood_g, 1000)
println("----------------------------------")

if converge
    C[4, 4] = iter
else  
    C[4, 4] = 1e8
end

# Froth
x_0 = [0.5, -2.0] 
println("Froth: ")
x, norma, converge, iter = descent(opt, x_0, froth, froth_g,1000)
println("----------------------------------")

if converge
    C[4, 5] = iter
else  
    C[4, 5] = 10e8
end

# Tasa de desempeño relativa.
r = zeros(4, 5)
for p in 1:5
    c_min = minimum(C[:,p])
    for s in 1:4
        r[s, p] = C[s, p] / c_min
    end 
end 

# Desempeño del algoritmo s, y tasa de desempeño relativa p.
function desempeño(r, s, t)
    fila = r[s, :]
    return count(x -> x <= t, fila) / length(fila)
end

# Plot de desempeño
t_vals = 1:250

gr()  
p = plot()

y_vals = [desempeño(r, 1, t) for t in t_vals]
plot!(p, t_vals, y_vals, label="Descenso paso constante")

y_vals = [desempeño(r, 2, t) for t in t_vals]
plot!(p, t_vals, y_vals, label="Armijo")

y_vals = [desempeño(r, 3, t) for t in t_vals]
plot!(p, t_vals, y_vals, label="Newton modificado + Armijo")

y_vals = [desempeño(r, 4, t) for t in t_vals]
plot!(p, t_vals, y_vals, label="Gradientes conjugados")

savefig(p, "desempeño.png")
