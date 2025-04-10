using Random, Distributions, LinearAlgebra, Bijectors

function compare_trained_and_untrained_flow(
    flow_trained::Bijectors.MultivariateTransformed,
    flow_untrained::Bijectors.MultivariateTransformed,
    true_dist::ContinuousMultivariateDistribution,
    n_samples::Int;
    kwargs...,
)
    samples_trained = rand(flow_trained, n_samples)
    samples_untrained = rand(flow_untrained, n_samples)
    samples_true = rand(true_dist, n_samples)

    p = scatter(
        samples_true[1, :],
        samples_true[2, :];
        label="True Distribution",
        color=:blue,
        markersize=2,
        alpha=0.5,
    )
    scatter!(
        p,
        samples_untrained[1, :],
        samples_untrained[2, :];
        label="Untrained Flow",
        color=:red,
        markersize=2,
        alpha=0.5,
    )
    scatter!(
        p,
        samples_trained[1, :],
        samples_trained[2, :];
        label="Trained Flow",
        color=:green,
        markersize=2,
        alpha=0.5,
    )
    plot!(; kwargs...)

    xlabel!(p, "X")
    ylabel!(p, "Y")
    title!(p, "Comparison of Trained and Untrained Flow")

    return p
end

# function check_trained_flow(
#     flow_trained::Bijectors.MultivariateTransformed,
#     true_dist::ContinuousMultivariateDistribution,
#     n_samples::Int;
#     kwargs...,
# )
#     samples_trained = rand_batch(flow_trained, n_samples)
#     samples_true = rand(true_dist, n_samples)

#     p = Plots.scatter(
#         samples_true[1, :],
#         samples_true[2, :];
#         label="True Distribution",
#         color=:green,
#         markersize=2,
#         alpha=0.5,
#     )
#     Plots.scatter!(
#         p,
#         samples_trained[1, :],
#         samples_trained[2, :];
#         label="Trained Flow",
#         color=:red,
#         markersize=2,
#         alpha=0.5,
#     )
#     Plots.plot!(; kwargs...)

#     Plots.title!(p, "Trained HamFlow")

#     return p
# end

function create_flow(Ls, q₀)
    ts =  reduce(∘, Ls)
    return transformed(q₀, ts)
end

function visualize(p::Bijectors.MultivariateTransformed, samples=rand(p, 1000))
    xrange = range(minimum(samples[1, :]) - 1, maximum(samples[1, :]) + 1; length=100)
    yrange = range(minimum(samples[2, :]) - 1, maximum(samples[2, :]) + 1; length=100)
    z = [exp(Distributions.logpdf(p, [x, y])) for x in xrange, y in yrange]
    fig = contour(xrange, yrange, z'; levels=15, color=:viridis, label="PDF", linewidth=2)
    scatter!(samples[1, :], samples[2, :]; label="Samples", alpha=0.3, legend=:bottomright)
    return fig
end
