using FiniteDifferences
using BSON: @save, @load
import Base:zero

extract(X, n, b) = reduce(vcat, (xt[n,b] for xt in X))
unzip(a) = (getfield.(a, x) for x in fieldnames(eltype(a)))
count_params(M) = sum(length(p) for p in Flux.params(M))

function fd_test_grads(loss, params; order=10)
    zygote_grads = gradient(loss, params)

    for (id, p) in enumerate(params)
        function fd_loss!(ptild)
            old_p = copy(p)
            p .= ptild  # This changes the param in model
            J = loss()
            p .= old_p  # return to original
            return J
        end

        fd_grad = grad(central_fdm(order, 1), fd_loss!, p)[1];
        error = norm(fd_grad - zygote_grads[p])
        println("Param ", id, " \tsize ", size(p), "\t gradient err = ", error)
    end
end

function estimate_lipschitz_lower(model; batches=1, seq_len=3000, plot_res=false, maxIter=300, step_size=1E-2, clip_at=1E-2, init_var=1E-3)

    # solver = PeacemanRachford(tol=1E-8, verbose=true)
    # (model::typeof(model))(x0, u) = model(x0, u, solver)

    x0 = init_state(model, batches)

    # u1 = collect(randn(model.nu, batches) for t in 1:seq_len)
    # u2 = collect(randn(model.nu, batches) for t in 1:seq_len)
    u1 = init_var * randn(seq_len, model.nu, batches)
    u2 = u1 .+ 1E-4 * randn(seq_len, model.nu, batches)

    unpack(ui) = [ui[t,:,:] for t in 1:seq_len]

    θ = Flux.Params([u1, u2, x0])

    opt = Flux.Optimiser(ClipValue(clip_at),
                        Flux.Optimise.ADAM(step_size),
                        Flux.Optimise.ExpDecay(1.0, 0.1, 200, 0.001)
                        )

    Lip() = norm(model(x0, unpack(u1))[1] - model(x0, unpack(u2))[1]) / (norm(u1 - u2))

    # Lip() = sum((model(x0, unpack(u1))[1] - model(x0, unpack(u2))[1]).^2) / sum((u1 - u2).^2)

    lips = []
    for iter in 1:maxIter
        L, back = Zygote.pullback(() -> -Lip(), θ)
        ∇L = back(one(L))
        update!(opt, θ, ∇L)
        append!(lips, L)
        println("Iter: ", iter, "\t L: ", -L, "\t η: ", opt[3].eta)
    end

    if plot_res == true

        du = unpack(u2) - unpack(u1)
        # Δu = 0.05 * du / norm(du)
        Δu = unpack(u2) - unpack(u1)
        yest1 = model(x0, unpack(u1))[1]
        yest2 = model(x0, unpack(u2))[1]
        # yest2 = model(x0, unpack(u1) + Δu)[1]
        Δy = yest1 - yest2
        
        data = (dy = Δy, du = Δu, Lip = norm(Δy) / norm(Δu),  u1 = unpack(u1), u2 = unpack(u2), yest1 = yest1, yest2 = yest2, model = model)
        @save "./results/sys_id/f16_adversarial_perturbations/ffren_10.bson" data


        p1 = plot(extract(Δy, 2, 1))
        plot!(extract(Δu, 1, 1))
        plot!(extract(Δu, 2, 1))
        plot(p1, p2, layout=(2, 1))
    end
    return maximum(-lips)
end


# return vectorized parameter list
function vec(param_list)
    return vcat((p[:] for p in param_list)...)
end

# Inverse operation of vec. Stores vectorized parameter into parameter list.
function store_vec!(param_list, θ)
    counter = 1
    for p in param_list
        θᵢ = θ[counter:counter + length(p) - 1]
        p .= reshape(θᵢ, size(p))
        counter += length(θᵢ)
    end
end
