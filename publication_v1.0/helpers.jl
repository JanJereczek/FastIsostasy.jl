function unpack(results::FastIsoProblem)
    Omega, p = reinit_structs_cpu(results.Omega, results.p)
    return Omega, results.c, p, results.t_out
end