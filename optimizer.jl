using LinearAlgebra

abstract type Optimizer end

# Optimizadores con búsqueda de línea (Armijo).
abstract type LineSearch <: Optimizer end

# Límites de seguridad para la regla de Armijo.
const ALPHA_MAX = 1e10
const ALPHA_MIN = 1e-16

# Descenso por gradiente ------------------
mutable struct GradientDescent <: Optimizer
    alpha::Float64
end

# Armijo -------------------------
mutable struct Armijo <: LineSearch
    alpha::Float64
    eps::Float64
    eta::Float64
end

# Newton -------------------------
mutable struct Newton <: LineSearch
    alpha::Float64
    eps::Float64
    eta::Float64
    gamma::Float64
end

# Gradientes conjugados (Fletcher-Reeves) ---------
mutable struct GradientesConjugados <: LineSearch
    alpha::Float64
    eps::Float64
    eta::Float64
end

# Resultado de una corrida.
struct Result
    x::Vector{Float64}
    norma::Float64
    converge::Bool
    iter::Int
end

# Regla de Armijo ----------------

# h(alpha) = f(x_k + alpha * d_k)
function h(f, x, alpha, d)
    return f(x + alpha*d)
end

# Chequea si cumple armijo.
function armijo_rule(f, x, alpha, eps, d, g)
    return h(f, x, alpha, d) <= f(x) + eps * dot(g, d) * alpha
end

function armijo_step(opt::LineSearch, f, x, d, gx)
    alpha = opt.alpha

    # Mientras armijo se cumpla multiplicamos por eta
    while armijo_rule(f, x, alpha, opt.eps, d, gx) && alpha < ALPHA_MAX
        alpha *= opt.eta
    end
    # Mientras armijo no se cumpla dividimos por eta
    while !armijo_rule(f, x, alpha, opt.eps, d, gx) && alpha > ALPHA_MIN
        alpha /= opt.eta
    end
    return alpha
end

# Dirección de descenso (una por método) ------------------
# Argumentos comunes: k = iteración actual, g = gradiente en x,
# g_prev y d_prev = gradiente y dirección anteriores, hess = Hessiana (o nothing).

direction(opt::GradientDescent, k, x, g, g_prev, d_prev, hess) = -g

direction(opt::Armijo, k, x, g, g_prev, d_prev, hess) = -g

# Newton modificado: si la Hessiana no es definida positiva, se desplaza el espectro.
function direction(opt::Newton, k, x, g, g_prev, d_prev, hess)
    B = hess(x)
    mu = eigvals(Symmetric(B))[1]
    if mu <= 0
        B = B + (-mu + opt.gamma)*I
    end
    return -(B \ g)
end

function direction(opt::GradientesConjugados, k, x, g, g_prev, d_prev, hess)
    # Reinicio en la primera iteración y cada n iteraciones.
    if k == 0 || k % length(x) == 0
        return -g
    end
    b = dot(g, g) / dot(g_prev, g_prev)
    d = -g + b * d_prev

    # Si no es dirección de descenso, reiniciamos.
    return dot(d, g) < 0 ? d : -g
end

# Tamaño de paso -----------------
stepsize(opt::GradientDescent, f, x, d, g) = opt.alpha
stepsize(opt::LineSearch, f, x, d, g) = armijo_step(opt, f, x, d, g)

# Descender ----------------------
function descent(opt::Optimizer, f, grad, hess, x_0, iterations::Int;
                 tol = 1e-8, verbose = true)
    x = copy(x_0)  # no muta el vector del caller
    g = grad(x)
    g_prev = nothing
    d = nothing
    iter = 0

    while iter < iterations && Inf > norm(g) > tol
        d = direction(opt, iter, x, g, g_prev, d, hess)
        t = stepsize(opt, f, x, d, g)
        x .+= t * d

        g_prev = g
        g = grad(x)
        iter += 1
    end

    norma = norm(g)
    converge = norma < tol
    res = Result(x, norma, converge, iter)
    verbose && report(res)
    return res
end

# Resultados
function report(res::Result)
    println("x*: $(res.x)")
    println("Norma del gradiente: $(res.norma)")
    println(res.converge ? "Convergió" : "No Convergió")
    println("Iteraciones: $(res.iter)")
end
