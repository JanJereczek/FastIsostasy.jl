
latexify(x) = [L"%$xi $\,$" for xi in x]
latexticks(x) = (x, latexify(x))
diagslice(X, N2, N4) = diag(X)[N2:N2+N4]

janjet = [:gray10, :cornflowerblue, :orange, :red3]
janjet_small = [:purple4, :royalblue, :cornflowerblue, :orange, :red3]