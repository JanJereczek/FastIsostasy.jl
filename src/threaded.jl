using .Threads

N = 1_000
A = rand(N, N)
B = zeros(N, N)
C = zeros(N, N)

function f!(X, A, j)
    X[:, j] = view(A, :, j) .* j
end

function serial!(X, A)
    for j in axes(A, 2)
        X[:, j] .= view(A, :, j) .* j
    end
end

function parall!(X, A)
    @threads for j in axes(A, 2)
        (@view X[:, j]) .= view(A, :, j) .* j
    end
end

serial!(B, A)
parall!(C, A)
B == C

@btime serial!($B, $A)
@btime parall!($C, $A)

