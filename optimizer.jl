using LinearAlgebra

abstract type Optimizer end

# Descenso por gradiente ------------------
mutable struct GradientDescent <: Optimizer
    alpha::Float64
end

# Un paso
function step!(x_k, grad, alpha) 
    x_k .-= alpha*grad(x_k)
end

# Descender
function descent(opt::GradientDescent, grad, x, iterations::Int; tol = 1e-8)
    x = copy(x)  # no muta el vector del caller
    iter = 0
    while (iter < iterations && Inf > norm(grad(x)) > tol)
        step!(x, grad, opt.alpha)
        iter += 1
    end
    converge = norm(grad(x)) < tol
    norma = norm(grad(x))
        
    # Resultados
    println("x*: $x")
    println("Norma del gradiente: $norma")
    if converge
        println("Convergió")
    else 
        println("No Convergió")
    end
    println("Iteraciones: $iter")
    return x, norma, converge, iter
end

# Armijo -------------------------
mutable struct Armijo <: Optimizer
    alpha::Float64    
    eps::Float64    
    eta::Float64    
end

# h(alpha) = f(x_k + alpha * d_k)
function h(f, x, alpha, d)
    return f(x + alpha*d)
end

# Chequea si cumple armijo.
function armijo_rule(f, x, alpha, eps, d, g) 
    return h(f, x, alpha, d) <= f(x) + eps * g' * d * alpha
end      

function armijo_step(opt::Optimizer, f, x, d, gx)
    alpha = opt.alpha          

    # Mientra armijo se cumpla multiplicamos por eta0
    while armijo_rule(f, x, alpha, opt.eps, d, gx)
        alpha *= opt.eta
    end
    # Mientras armijo no se cumple dividimos por eta
    while !armijo_rule(f, x, alpha, opt.eps, d, gx) 
        alpha /= opt.eta
    end
    return alpha
end

# Descenso armijo.
function descent(opt::Armijo, f, x, grad, iterations::Int; tol = 1e-8)
    x = copy(x)  
    iter = 0
    while (iter < iterations && Inf > norm(grad(x)) > tol)
        alpha = armijo_step(opt, f, x, -grad(x), grad(x))
        step!(x, grad, alpha)
        iter += 1
    end
    converge = norm(grad(x)) < tol
    norma = norm(grad(x))
        
    # Resultados
    println("x*: $x")
    println("Norma del gradiente: $norma")
    if converge
        println("Convergió")
    else 
        println("No Convergió")
    end
    println("Iteraciones: $iter")
    return x, norma, converge, iter
end

# Newton -------------------------
mutable struct Newton <: Optimizer
    alpha::Float64
    eps::Float64
    eta::Float64
    gamma::Float64
end

function descent(opt::Newton, x_0, f, grad_f, Hf, iterations::Int; tol = 1e-8)
    x_k = copy(x_0)
    iter = 0 

    while Inf > norm(grad_f(x_k)) > tol && iter < iterations

        B = Hf(x_k)
        mu = eigvals(Symmetric(B))[1]  
        
        if mu <= 0
            B = B + (-mu+ opt.gamma)I
        end
        d_k = -B\grad_f(x_k)
        t_k = armijo_step(opt, f, x_k, d_k, grad_f(x_k))
        x_k .+= t_k*d_k
        iter += 1
    end
    converge = norm(grad_f(x_k)) < tol
    norma = norm(grad_f(x_k))
        
    # Resultados
    println("x*: $(x_k)")
    println("Norma del gradiente: $norma")
    if converge
        println("Convergió")
    else 
        println("No Convergió")
    end
    println("Iteraciones: $iter")

    return x_k, norma, converge, iter
end

# Quasi Newton --------------
mutable struct GradientesConjugados <: Optimizer
    alpha::Float64    
    eps::Float64    
    eta::Float64 
end

function descent(opt::GradientesConjugados, x_0, f, grad_f, iterations::Int; tol = 1e-8)
    # Copia inicial
    x_k = copy(x_0)

    # Dirección inicial
    d_k = -grad_f(x_k)

    # Primer paso con Armijo
    t_k = armijo_step(opt, f, x_k, d_k, grad_f(x_k))
    x_k .+= t_k * d_k

    # Inicialización
    iter = 1

    while norm(grad_f(x_k)) > tol && iter < iterations
        
        if iter%length(x_0) != 0
            b_k = norm(grad_f(x_k))^2 / norm(grad_f(x_k - t_k*d_k))^2
            d_k = -grad_f(x_k) + b_k * d_k
        else
            d_k = -grad_f(x_k)
        end 

        # Paso Armijo
        t_k = armijo_step(opt, f, x_k, d_k, grad_f(x_k))

        # Actualizar punto
        x_k .+= t_k * d_k

        iter += 1
    end

    # Resultados
    norma = norm(grad_f(x_k))
    converge = norma < tol

    println("x*: $(x_k)")
    println("Norma del gradiente: $norma")
    println(converge ? "Convergió" : "No Convergió")
    println("Iteraciones: $iter")

    return x_k, norma, converge, iter
end

