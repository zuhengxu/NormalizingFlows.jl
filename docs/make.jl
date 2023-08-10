using NormalizingFlows
using Documenter

DocMeta.setdocmeta!(
    NormalizingFlows, :DocTestSetup, :(using NormalizingFlows); recursive=true
)

makedocs(;
    modules=[NormalizingFlows],
    repo="https://github.com/TuringLang/NormalizingFlows.jl/blob/{commit}{path}#{line}",
    sitename="NormalizingFlows.jl",
    format=Documenter.HTML(),
    pages=[
        "Home" => "index.md",
        "API" => "api.md",
        "Examples" => "example.md",
        "Customize your own flow layer" => "custommized_layer.md",
    ],
)

deploydocs(; repo="github.com/TuringLang/NormalizingFlows.jl", devbranch="main")
