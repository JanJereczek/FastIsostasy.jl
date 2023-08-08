function unpack(results::FastIso)
    Omega, p = reinit_structs_cpu(results.Omega, results.p)
    return Omega, results.c, p, results.t_out
end