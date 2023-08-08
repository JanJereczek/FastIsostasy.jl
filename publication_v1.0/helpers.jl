function unpack(results::FastIso)
    Omega, p = copystructs2cpu(results.Omega, results.p)
    return Omega, results.c, p, results.t_out
end