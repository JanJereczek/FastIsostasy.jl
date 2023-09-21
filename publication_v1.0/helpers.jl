function unpack(results::FastIsoProblem)
    Omega, p = reinit_structs_cpu(results.Omega, results.p)
    return Omega, results.c, p, results.t_out
end

not(x::Bool) = !x
latexify(x) = [L"%$xi $\,$" for xi in x]
latexticks(x) = (x, latexify(x))
diagslice(X, N2, N4) = diag(X)[N2:N2+N4]

# janjet = [:purple4, :purple1, :royalblue, :cornflowerblue, :orange, :red3]
# janjet = [:purple4, :purple1, :limegreen, :green, :royalblue, :cornflowerblue, :orange, :red3]
# janjet = [:purple4, :purple1, :orchid, :cornflowerblue, :royalblue, :orange, :red3]
janjet = [:gray10, :cornflowerblue, :orange, :red3]
janjet_small = [:purple4, :royalblue, :cornflowerblue, :orange, :red3]