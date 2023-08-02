
function rrule(::typeof(_rollout!), env::AbstractMJEnv, xt::AbstractMatrix{T}, ut::AbstractMatrix{T}) where T

    # Forward pass
    out = _rollout!(env, xt, ut)

    # Write pullback
    project_x = ProjectTo(xt)
    project_u = ProjectTo(ut)
    function _rollout_pullback(ȳ)

        (x̄s, ȳs) = ȳ

        # No gradient for function itself or environment
        f̄ = NoTangent()
        ēnv = NoTangent()

        # Compute Jacobians for each sample in batch
        # TODO: Multi-thread this when we get Julia bindings for MuJoCo.
        N = size(xt, 2)
        x̄t = zeros(T, env.nx, N)
        ūt = zeros(T, env.nu, N)
        for i in 1:N
            xi = @view xt[:,i]
            ui = @view ut[:,i]
            A, B, C, D = get_jacobians!(env, xi, ui)

            x̄i = @view x̄s[:,i]
            env.sense && (ȳi = @view ȳs[:,i])
            x̄t[:,i] = env.sense ? A' * x̄i + C' * ȳi : A' * x̄i
            ūt[:,i] = env.sense ? B' * x̄i + D' * ȳi : B' * x̄i 
        end
        return f̄, ēnv, project_x(x̄t), project_u(ūt)
    end
    return out, _rollout_pullback
end

@non_differentiable get_state(env::AbstractMJEnv)