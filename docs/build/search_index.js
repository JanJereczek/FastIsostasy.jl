var documenterSearchIndex = {"docs":
[{"location":"examples.html#Examples","page":"Examples","title":"Examples","text":"","category":"section"},{"location":"examples.html#Multi-layer-Earth","page":"Examples","title":"Multi-layer Earth","text":"","category":"section"},{"location":"examples.html","page":"Examples","title":"Examples","text":"FastIsostasy relies on a (polar) stereographic projection that allows to treat the radially-layered, onion-like structure of the solid Earth as a superposition of horizontal layers. Furthermore, FastIsostasy reduces this 3D problem into a 2D problem by collapsing the depth dimension by computing an effective viscosity field. The user is required to provide the 3D information, which will then be used under the hood to compute the effective viscosity. This tutorial shows such an example.","category":"page"},{"location":"examples.html","page":"Examples","title":"Examples","text":"We want to render a situation similar to the one depicted below:","category":"page"},{"location":"examples.html","page":"Examples","title":"Examples","text":"(Image: Schematic representation of the three-layer set-up.)","category":"page"},{"location":"examples.html","page":"Examples","title":"Examples","text":"Initializing a MultilayerEarth with parameters corresponding to this situation automatically computes the conversion from a 3D to a 2D problem. This can be simply executed by running:","category":"page"},{"location":"examples.html","page":"Examples","title":"Examples","text":"W = 3000e3      # (m) half-width of the domain\nn = 8           # implies an NxN grid with N = 2^n = 256.\nOmega = ComputationDomain(W, n)\nc = PhysicalConstants()\n\nlv = [1e19, 1e21]       # (Pa*s)\nlb = [88e3, 400e3]      # (m)\np = MultilayerEarth(Omega, c, layer_viscosities = lv, layer_boundaries = lb)","category":"page"},{"location":"examples.html","page":"Examples","title":"Examples","text":"The next section shows how to use the now obtained p::MultilayerEarth for actual GIA computation.","category":"page"},{"location":"examples.html#Simple-load-and-geometry","page":"Examples","title":"Simple load and geometry","text":"","category":"section"},{"location":"examples.html","page":"Examples","title":"Examples","text":"We now apply a constant load, here a cylinder of ice with radius R = 1000 km and thickness H = 1 km, over the domain introduced in Multi-layer Earth. To obtain the bedrock displacement over time and store it at time steps specified by a vector t_out, we can use the convenience function fastisostasy:","category":"page"},{"location":"examples.html","page":"Examples","title":"Examples","text":"R = 1000e3                  # ice disc radius (m)\nH = 1000.0                  # ice disc thickness (m)\nHice = uniform_ice_cylinder(Omega, R, H)\nt_out = years2seconds.([0.0, 100.0, 500.0, 1500.0, 5000.0, 10_000.0, 50_000.0])\n\nresults = fastisostasy(t_out, Omega, c, p, Hice)","category":"page"},{"location":"examples.html","page":"Examples","title":"Examples","text":"Yes, that was it! You can now easily access the elastic and viscous displacement by calling results.elastic or results.viscous. For the present case, the latter can be compared to an analytic solution that is known for this particular case. Let's look at the accuracy of our numerical scheme over time by running following plotting commands:","category":"page"},{"location":"examples.html","page":"Examples","title":"Examples","text":"M = Omega.N ÷ 2\nx = diag(Omega.X)[1:M]\ny = diag(Omega.Y)[1:M]\nr = sqrt.( x .^ 2 + y .^ 2 )\n\nfig = Figure()\nax3 = Axis(fig[1, 1])\ncolors = [:gray80, :gray65, :gray50, :gray35, :gray20, :gray5]\n\nfor i in eachindex(t_out)\n    t = t_out[i]\n    analytic_solution_r(r) = analytic_solution(r, t, c, p, H, R, analytic_support)\n    u_analytic = analytic_solution_r.( r )\n    u_numeric = diag(results.viscous[i])\n    lines!(ax3, x, u_numeric[1:M], color = colors[i], linewidth = 5)\n    lines!(ax3, x, u_analytic, color = colors[i], linewidth = 5, linestyle = :dash)\nend\nfig","category":"page"},{"location":"examples.html#Time-changing-load","page":"Examples","title":"Time-changing load","text":"","category":"section"},{"location":"examples.html","page":"Examples","title":"Examples","text":"That looks pretty good! One might however object that the convenience function fastisostasy ends up being not so convenient as soon as the ice load changes over time. This case can however be easily handled by providing snapshots of the ice thickness and their assoicated time. By passing this to fastisostasy, an interpolator is created and called within the time integration. Let's create a tool example where the thickness of the ice cylinder asymptotically grows from 0 to 1 km, this can be implemented by:","category":"page"},{"location":"examples.html","page":"Examples","title":"Examples","text":"normalized_asymptote(t) = 1 - exp(-t)\nt_Hice_asymptotic = 0:10.0:t_out[end]\nHice_asymptotic = [normalized_asymptote(t) .* Hice for t in t_Hice_asymptote]\nresults = fastisostasy(t_out, Omega, c, p, t_Hice_asymptotic, Hice_asymptotic)","category":"page"},{"location":"examples.html","page":"Examples","title":"Examples","text":"This concept will also apply to the upper-mantle viscosity in future versions, as it can change over time.","category":"page"},{"location":"examples.html#GPU-support","page":"Examples","title":"GPU support","text":"","category":"section"},{"location":"examples.html","page":"Examples","title":"Examples","text":"For about n  6, the previous example can be computed even faster by using GPU parallelism. It could not represent less work from the user's perspective, as it boils down to calling the ComputationDomain with an extra keyword argument:","category":"page"},{"location":"examples.html","page":"Examples","title":"Examples","text":"Omega = ComputationDomain(W, n, use_cuda=true)","category":"page"},{"location":"examples.html","page":"Examples","title":"Examples","text":"That's it, nothing more! One could suggest you lay back but your computation might be completed too soon for that.","category":"page"},{"location":"examples.html","page":"Examples","title":"Examples","text":"info: Only CUDA supported!\nFor now only Nvidia GPUs are supported (sorry Mac users, destiny is taking its revenge upon you for being so cool) and there is no plan of extending this compatibility at this point.","category":"page"},{"location":"examples.html#Simple-load-and-geometry-DIY","page":"Examples","title":"Simple load and geometry - DIY","text":"","category":"section"},{"location":"examples.html","page":"Examples","title":"Examples","text":"Nonetheless, as any high-level convenience function, fastisostasy has limitations. An ice-sheet modeller typically wants to embed FastIsostasy within a time-stepping loop. This can be easily done by getting familiar with some intermediate-level functions:","category":"page"},{"location":"examples.html","page":"Examples","title":"Examples","text":"W = 3000e3      # (m) half-width of the domain\nn = 8           # implies an NxN grid with N = 2^n = 256.\nOmega = ComputationDomain(W, n)\nc = PhysicalConstants()\np = MultilayerEarth(Omega, c)\n\nR = 1000e3                  # ice disc radius (m)\nH = 1000.0                  # ice disc thickness (m)\nHice = uniform_ice_cylinder(Omega, R, H)\nt_out = years2seconds.([0.0, 100.0, 500.0, 1500.0, 5000.0, 10_000.0, 50_000.0])\n\nresults = fastisostasy(t_out, Omega, c, p, Hice)","category":"page"},{"location":"examples.html#GIA-following-Antarctic-deglaciation","page":"Examples","title":"GIA following Antarctic deglaciation","text":"","category":"section"},{"location":"examples.html","page":"Examples","title":"Examples","text":"We now want to provide a tough example that presents:","category":"page"},{"location":"examples.html","page":"Examples","title":"Examples","text":"a heterogeneous lithosphere thickness\na heterogeneous upper-mantle viscosity\nvarious viscous channels\na more elaborate load that evolves over time\nchanges in the sea-level","category":"page"},{"location":"examples.html","page":"Examples","title":"Examples","text":"For this we run a deglaciation of Antarctica, based on the ice thickness estimated in GLAC1D.","category":"page"},{"location":"examples.html","page":"Examples","title":"Examples","text":"W = 3000e3      # (m) half-width of the domain\nn = 8           # implies an NxN grid with N = 2^n = 256.\nOmega = ComputationDomain(W, n)\nc = PhysicalConstants()","category":"page"},{"location":"examples.html#Inversion-of-solid-Earth-parameters","page":"Examples","title":"Inversion of solid-Earth parameters","text":"","category":"section"},{"location":"examples.html","page":"Examples","title":"Examples","text":"FastIsostasy.jl relies on simplification of the full problem and might therefore need a calibration step to match the output of a 3D GIA model. By means of an unscented Kalman inversion, one can e.g. infer the appropriate effective upper-mantle viscosity based on the response of a 3D GIA model to a given load. Whereas this is know to be a tedious step, FastIsostasy is developped to ease the procedure by providing a convenience struct Paraminversion that can be run by:","category":"page"},{"location":"examples.html","page":"Examples","title":"Examples","text":"L = T(3000e3)               # half-length of the square domain (m)\nOmega = ComputationDomain(L, n)\nc = PhysicalConstants()\n\nlb = [88e3, 180e3, 280e3, 400e3]\nlv = get_wiens_layervisc(Omega)\np = MultilayerEarth(\n    Omega,\n    c,\n    layer_boundaries = lb,\n    layer_viscosities = lv,\n)\nground_truth = copy(p.effective_viscosity)\n\nR = T(2000e3)               # ice disc radius (m)\nH = T(1000)                 # ice disc thickness (m)\nHcylinder = uniform_ice_cylinder(Omega, R, H)\nt_out = years2seconds.(0.0:1_000.0:2_000.0)\n\nt1 = time()\nresults = fastisostasy(t_out, Omega, c, p, Hcylinder, ODEsolver=BS3(), interactive_sealevel=false)\nt_fastiso = time() - t1\nprintln(\"Took $t_fastiso seconds!\")\nprintln(\"-------------------------------------\")\n\ntinv = t_out[2:end]\nHice = [Hcylinder for t in tinv]\nY = results.viscous[2:end]\nparaminv = ParamInversion(Omega, c, p, tinv, Y, Hice)\npriors, ukiobj = perform(paraminv)\nlogeta, Gx, e_mean, e_sort = extract_inversion(priors, ukiobj, paraminv)","category":"page"},{"location":"collected_garbage.html","page":"-","title":"-","text":"<!– ","category":"page"},{"location":"collected_garbage.html","page":"-","title":"-","text":"Inferring the lithospheric thickness and the upper-mantle viscosity is all but trivial since ice sheets famously present difficult access to scientific missions, leading to a lack of seismologic stations compared to other regions of the world. The presence of few kilometers of ice between most stations and the bedrock is a further challenge. Although major advances have been recently made, Antarctica is therefore the region where solid-Earth parameters are worst-constrained.","category":"page"},{"location":"collected_garbage.html","page":"-","title":"-","text":"[ 1, 2 ].","category":"page"},{"location":"collected_garbage.html","page":"-","title":"-","text":"Compared to [1, 2], FastIsostasy.jl does not assume constant fields for parameters of the solid Earth. It thus offers an open-source and performant generalization of the original articles.  This allows to transform the PDE describing the physics into an ODE and accelerate the computation, mainly due to the highly optimized functions available for fast-fourier transform (FFT).","category":"page"},{"location":"collected_garbage.html","page":"-","title":"-","text":"Computing the vertical displacement of the bedrock can be computed much more efficiently by relying on the precomputation of some terms and operations. In FastIsostasy.jl, this can be easily performed:","category":"page"},{"location":"collected_garbage.html","page":"-","title":"-","text":"–>","category":"page"},{"location":"collected_garbage.html","page":"-","title":"-","text":"<!– ","category":"page"},{"location":"collected_garbage.html#A-three-layer-model","page":"-","title":"A three-layer model","text":"","category":"section"},{"location":"collected_garbage.html","page":"-","title":"-","text":"Let x, y be the coordinates spanning the projection of the Earth surface and z the depth coordinate. The present model assumes three layers over the z-dimension:","category":"page"},{"location":"collected_garbage.html","page":"-","title":"-","text":"The elastic lithosphere.\nA channel representing the upper mantle, usually displaying strong variance of viscosity over x and y.\nA half-space representing the rest of the mantle, usually with small variance of viscosity over x and y.","category":"page"},{"location":"collected_garbage.html","page":"-","title":"-","text":"The two-layer model is a special case of this and can be obtained by setting the channel parameters to be the same as the ones of the half space.","category":"page"},{"location":"collected_garbage.html","page":"-","title":"-","text":"(Image: Schematic representation of the three-layer model) –>","category":"page"},{"location":"collected_garbage.html#Arrays","page":"-","title":"Arrays","text":"","category":"section"},{"location":"collected_garbage.html","page":"-","title":"-","text":"Array dimensions correspond to the spatial dimension of the variable they describe. If they evolve over time, they are stored as vector of arrays. For instance, the vertical displacement of the bedrock is a 2D variable that evolves over time. Therefore, it is stored in a Vector{Matrix}.","category":"page"},{"location":"APIref.html#API-reference","page":"API reference","title":"API reference","text":"","category":"section"},{"location":"APIref.html#Basic-structs","page":"API reference","title":"Basic structs","text":"","category":"section"},{"location":"APIref.html","page":"API reference","title":"API reference","text":"ComputationDomain\nPhysicalConstants\nMultilayerEarth\nRefSealevelState\nSealevelState\nPrecomputedFastiso\nSuperStruct\nFastisoResults","category":"page"},{"location":"APIref.html#FastIsostasy.ComputationDomain","page":"API reference","title":"FastIsostasy.ComputationDomain","text":"ComputationDomain\n\nReturn a struct containing all information related to geometry of the domain and potentially used parallelism.\n\n\n\n\n\n","category":"type"},{"location":"APIref.html#FastIsostasy.PhysicalConstants","page":"API reference","title":"FastIsostasy.PhysicalConstants","text":"PhysicalConstants\n\nReturn a struct containing important physical constants.\n\n\n\n\n\n","category":"type"},{"location":"APIref.html#FastIsostasy.MultilayerEarth","page":"API reference","title":"FastIsostasy.MultilayerEarth","text":"MultilayerEarth\n\nReturn a struct containing all information related to the radially layered structure of the solid Earth and its parameters.\n\n\n\n\n\n","category":"type"},{"location":"APIref.html#FastIsostasy.SealevelState","page":"API reference","title":"FastIsostasy.SealevelState","text":"SealevelState\n\nReturn a mutable struct containing the slstate which will be updated over the simulation.\n\n\n\n\n\n","category":"type"},{"location":"APIref.html#FastIsostasy.PrecomputedFastiso","page":"API reference","title":"FastIsostasy.PrecomputedFastiso","text":"PrecomputedFastiso\n\nReturn a struct containing all the pre-computed terms needed for the forward integration of the model.\n\n\n\n\n\n","category":"type"},{"location":"APIref.html#FastIsostasy.SuperStruct","page":"API reference","title":"FastIsostasy.SuperStruct","text":"SuperStruct\n\nReturn a struct containing all the other structs needed for the forward integration of the model.\n\n\n\n\n\n","category":"type"},{"location":"APIref.html#Utils","page":"API reference","title":"Utils","text":"","category":"section"},{"location":"APIref.html","page":"API reference","title":"API reference","text":"years2seconds\nseconds2years\nm_per_sec2mm_per_yr\nmatrify_vectorconstant\nmatrify_constant\nget_r\nmeshgrid\ndist2angulardist\nsphericaldistance\nscalefactor\nlatlon2stereo\nstereo2latlon\nget_rigidity\nget_effective_viscosity\nget_differential_fourier\nget_viscosity_ratio\nthree_layer_scaling\nloginterp_viscosity\nhyperbolic_channel_coeffs\nget_greenintegrand_coeffs\nbuild_greenintegrand\nget_elasticgreen\nget_quad_coeffs\nquadrature1D\nquadrature2D\nget_normalized_lin_transform\nnormalized_lin_transform\nkernelpromote","category":"page"},{"location":"APIref.html#FastIsostasy.years2seconds","page":"API reference","title":"FastIsostasy.years2seconds","text":"years2seconds(t)\n\nConvert input time t from years to seconds.\n\n\n\n\n\n","category":"function"},{"location":"APIref.html#FastIsostasy.seconds2years","page":"API reference","title":"FastIsostasy.seconds2years","text":"seconds2years(t)\n\nConvert input time t from seconds to years.\n\n\n\n\n\n","category":"function"},{"location":"APIref.html#FastIsostasy.m_per_sec2mm_per_yr","page":"API reference","title":"FastIsostasy.m_per_sec2mm_per_yr","text":"m_per_sec2mm_per_yr(dudt)\n\nConvert displacement rate dudt from \n\n\n\n\n\n","category":"function"},{"location":"APIref.html#FastIsostasy.matrify_vectorconstant","page":"API reference","title":"FastIsostasy.matrify_vectorconstant","text":"matrify_vectorconstant(x, N)\n\nGenerate a vector of constant matrices from a vector of constants.\n\n\n\n\n\n","category":"function"},{"location":"APIref.html#FastIsostasy.get_r","page":"API reference","title":"FastIsostasy.get_r","text":"get_r(x, y)\n\nGet euclidean distance of point (x, y) to origin.\n\n\n\n\n\n","category":"function"},{"location":"APIref.html#FastIsostasy.meshgrid","page":"API reference","title":"FastIsostasy.meshgrid","text":"meshgrid(x, y)\n\nReturn a 2D meshgrid spanned by x, y.\n\n\n\n\n\n","category":"function"},{"location":"APIref.html#FastIsostasy.dist2angulardist","page":"API reference","title":"FastIsostasy.dist2angulardist","text":"dist2angulardist()\n\n\n\n\n\n","category":"function"},{"location":"APIref.html#FastIsostasy.latlon2stereo","page":"API reference","title":"FastIsostasy.latlon2stereo","text":"latlon2stereo(lat, lon, lat0, lon0)\n\nConvert latitude-longitude coordinates to stereographically projected (x,y). Reference: John P. Snyder (1987), p. 157, eq. (21-2), (21-3), (21-4).\n\n\n\n\n\n","category":"function"},{"location":"APIref.html#FastIsostasy.stereo2latlon","page":"API reference","title":"FastIsostasy.stereo2latlon","text":"stereo2latlon(x, y, lat0, lon0)\n\nConvert stereographic (x,y)-coordinates to latitude-longitude. Reference: John P. Snyder (1987), p. 159, eq. (20-14), (20-15), (20-18), (21-15).\n\n\n\n\n\n","category":"function"},{"location":"APIref.html#FastIsostasy.get_rigidity","page":"API reference","title":"FastIsostasy.get_rigidity","text":"get_rigidity(t::T, E::T, nu::T) where {T<:AbstractFloat}\n\nCompute rigidity based on thickness t, Young modulus E and Poisson ration nu\n\n\n\n\n\n","category":"function"},{"location":"APIref.html#FastIsostasy.loginterp_viscosity","page":"API reference","title":"FastIsostasy.loginterp_viscosity","text":"loginterp_viscosity(tvec, layer_viscosities, layers_thickness, pseudodiff)\n\nCompute a log-interpolator of the equivalent viscosity from provided viscosity fields layer_viscosities at time stamps tvec.\n\n\n\n\n\n","category":"function"},{"location":"APIref.html#FastIsostasy.get_elasticgreen","page":"API reference","title":"FastIsostasy.get_elasticgreen","text":"get_elasticgreen(Omega, quad_support, quad_coeffs)\n\nIntegrate load response over field by using 2D quadrature with specified support points and associated coefficients.\n\n\n\n\n\n","category":"function"},{"location":"APIref.html#FastIsostasy.get_quad_coeffs","page":"API reference","title":"FastIsostasy.get_quad_coeffs","text":"get_quad_coeffs(T, n)\n\nReturn support points and associated coefficients with specified Type for Gauss-Legendre quadrature.\n\n\n\n\n\n","category":"function"},{"location":"APIref.html#FastIsostasy.quadrature1D","page":"API reference","title":"FastIsostasy.quadrature1D","text":"quadrature1D(f, n, x1, x2)\n\nCompute 1D Gauss-Legendre quadrature of f between x1 and x2 based on n support points.\n\n\n\n\n\n","category":"function"},{"location":"APIref.html#FastIsostasy.kernelpromote","page":"API reference","title":"FastIsostasy.kernelpromote","text":"kernelpromote(X, arraykernel)\n\nPromote X to the kernel (Array or CuArray) specified by arraykernel.\n\n\n\n\n\n","category":"function"},{"location":"APIref.html#Mechanics","page":"API reference","title":"Mechanics","text":"","category":"section"},{"location":"APIref.html","page":"API reference","title":"API reference","text":"ice_load\nfastisostasy\ninit_superstruct\nforward_isostasy\ninit_results\nforwardstep_isostasy!\ndudt_isostasy!\nsimple_euler!\napply_bc\ncompute_elastic_response","category":"page"},{"location":"APIref.html#FastIsostasy.ice_load","page":"API reference","title":"FastIsostasy.ice_load","text":"ice_load(c::PhysicalConstants{T}, H::T)\n\nCompute ice load based on ice thickness.\n\n\n\n\n\n","category":"function"},{"location":"APIref.html#FastIsostasy.fastisostasy","page":"API reference","title":"FastIsostasy.fastisostasy","text":"fastisostasy()\n\nMain function. List of all available solvers here.\n\n\n\n\n\n","category":"function"},{"location":"APIref.html#FastIsostasy.init_superstruct","page":"API reference","title":"FastIsostasy.init_superstruct","text":"init_superstruct()\n\nInit a SuperStruct containing all necessary fields for forward integration in time.\n\n\n\n\n\n","category":"function"},{"location":"APIref.html#FastIsostasy.dudt_isostasy!","page":"API reference","title":"FastIsostasy.dudt_isostasy!","text":"dudt_isostasy!()\n\nUpdate the displacement rate dudt.\n\n\n\n\n\n","category":"function"},{"location":"APIref.html#Sea-level","page":"API reference","title":"Sea-level","text":"","category":"section"},{"location":"APIref.html","page":"API reference","title":"API reference","text":"update_slstate!\nupdate_geoid!\nget_loadchange\nget_geoidgreen\nupdate_loadcolumns!\nupdate_sealevel!\nupdate_slc!\nupdate_V_af!\nupdate_slc_af!\nupdate_V_pov!\nupdate_slc_pov!\nupdate_V_den!\nupdate_slc_den!","category":"page"},{"location":"APIref.html#FastIsostasy.update_slstate!","page":"API reference","title":"FastIsostasy.update_slstate!","text":"update_slstate!(sstruct::SuperStruct, u::Matrix, H_ice::Matrix)\n\nUpdate the ::SealevelState computing the current geoid perturbation, the sea-level changes and the load columns for the next time step of the isostasy integration.\n\n\n\n\n\n","category":"function"},{"location":"APIref.html#FastIsostasy.update_geoid!","page":"API reference","title":"FastIsostasy.update_geoid!","text":"update_geoid!(sstruct::SuperStruct)\n\nUpdate the geoid of a ::SealevelState by convoluting the Green's function with the load change.\n\n\n\n\n\n","category":"function"},{"location":"APIref.html#FastIsostasy.get_loadchange","page":"API reference","title":"FastIsostasy.get_loadchange","text":"get_loadchange(sstruct::SuperStruct)\n\nCompute the load change compared to the reference configuration after Coulon et al. 2021. Difference to Coulon: HERE WE SCALE WITH CORRECTION FACTOR FOR PROJECTION!\n\n\n\n\n\n","category":"function"},{"location":"APIref.html#FastIsostasy.get_geoidgreen","page":"API reference","title":"FastIsostasy.get_geoidgreen","text":"get_geoidgreen(sstruct::SuperStruct)\n\nReturn the Green's function used to compute the changes in geoid.\n\nReference\n\nCoulon et al. 2021.\n\n\n\n\n\n","category":"function"},{"location":"APIref.html#FastIsostasy.update_loadcolumns!","page":"API reference","title":"FastIsostasy.update_loadcolumns!","text":"update_loadcolumns!(sstruct::SuperStruct, u::XMatrix, H_ice::XMatrix)\n\nUpdate the load columns of a ::SealevelState.\n\n\n\n\n\n","category":"function"},{"location":"APIref.html#FastIsostasy.update_sealevel!","page":"API reference","title":"FastIsostasy.update_sealevel!","text":"update_sealevel!(sstruct::SuperStruct)\n\nUpdate the sea-level ::SealevelState by adding the various contributions.\n\nReference\n\nCoulon et al. (2021), Figure 1.\n\n\n\n\n\n","category":"function"},{"location":"APIref.html#FastIsostasy.update_slc!","page":"API reference","title":"FastIsostasy.update_slc!","text":"update_slc!(sstruct::SuperStruct)\n\nUpdate the sea-level contribution of melting above floatation, density correction and potential ocean volume.\n\nReference\n\nGoelzer et al. (2020), eq. (12).\n\n\n\n\n\n","category":"function"},{"location":"APIref.html#FastIsostasy.update_V_af!","page":"API reference","title":"FastIsostasy.update_V_af!","text":"update_V_af!(sstruct::SuperStruct)\n\nUpdate the ice volume above floatation.\n\nReference\n\nGoelzer et al. (2020), eq. (13). Note: we do not use eq. (1) as it is only a special case of eq. (13) that does not allow a correct representation of external sea-level forcings.\n\n\n\n\n\n","category":"function"},{"location":"APIref.html#FastIsostasy.update_slc_af!","page":"API reference","title":"FastIsostasy.update_slc_af!","text":"update_slc_af!(sstruct::SuperStruct)\n\nUpdate the sea-level contribution of ice above floatation.\n\nReference\n\nGoelzer et al. (2020), eq. (2).\n\n\n\n\n\n","category":"function"},{"location":"APIref.html#FastIsostasy.update_V_pov!","page":"API reference","title":"FastIsostasy.update_V_pov!","text":"update_V_pov!(sstruct::SuperStruct)\n\nUpdate the potential ocean volume.\n\nReference\n\nGoelzer et al. (2020), eq. (14). Note: we do not use eq. (8) as it is only a special case of eq. (14) that does not allow a correct representation of external sea-level forcinsstruct.slstate.\n\n\n\n\n\n","category":"function"},{"location":"APIref.html#FastIsostasy.update_slc_pov!","page":"API reference","title":"FastIsostasy.update_slc_pov!","text":"update_slc_pov!(sstruct::SuperStruct)\n\nUpdate the sea-level contribution associated with the potential ocean volume.\n\nReference\n\nGoelzer et al. (2020), eq. (9).\n\n\n\n\n\n","category":"function"},{"location":"APIref.html#FastIsostasy.update_V_den!","page":"API reference","title":"FastIsostasy.update_V_den!","text":"update_V_den!(sstruct::SuperStruct)\n\nUpdate the ocean volume associated with the density correction.\n\nReference\n\nGoelzer et al. (2020), eq. (10).\n\n\n\n\n\n","category":"function"},{"location":"APIref.html#FastIsostasy.update_slc_den!","page":"API reference","title":"FastIsostasy.update_slc_den!","text":"update_slc_den!(sstruct::SuperStruct)\n\nUpdate the sea-level contribution associated with the density correction.\n\nReference\n\nGoelzer et al. (2020), eq. (11).\n\n\n\n\n\n","category":"function"},{"location":"APIref.html#Parameter-inversion","page":"API reference","title":"Parameter inversion","text":"","category":"section"},{"location":"index.html#Introduction","page":"Introduction","title":"Introduction","text":"","category":"section"},{"location":"index.html#Getting-started","page":"Introduction","title":"Getting started","text":"","category":"section"},{"location":"index.html","page":"Introduction","title":"Introduction","text":"FastIsostasy.jl is work under devlopment and is not a registered julia package so far. To install it, please run:","category":"page"},{"location":"index.html","page":"Introduction","title":"Introduction","text":"using Pkg\nPkg.add(\"https://github.com/JanJereczek/FastIsostasy.jl\")","category":"page"},{"location":"index.html#FastIsostasy.jl-–-For-whom?","page":"Introduction","title":"FastIsostasy.jl – For whom?","text":"","category":"section"},{"location":"index.html","page":"Introduction","title":"Introduction","text":"This package is mainly addressed to ice sheet modellers looking for a regional model of glacial isostatic adjustment (GIA) that (1) captures the 3D structure of solid-Earth parameters, (2) computes an approximation of the sea-level equation, (3) runs kiloyear simulations on high resolution within minutes (without the need of HPC hardware) and (4) comes with ready-to-use calibration tools. For GIA \"purists\", this package is likely to miss interesting processes but we belive that the ridiculous run-time of FastIsostasy.jl can help them to perform some fast prototypting of a problem they might then transfer to a more comprehensive model.","category":"page"},{"location":"index.html","page":"Introduction","title":"Introduction","text":"tip: Star us on GitHub!\nIf you have found this library useful, please consider starring it on GitHub. This gives us a lower bound of the satisfied user count.","category":"page"},{"location":"index.html#How-to-read-the-docs?","page":"Introduction","title":"How to read the docs?","text":"","category":"section"},{"location":"index.html","page":"Introduction","title":"Introduction","text":"If you already know about GIA, skip to Overview of GIA for ice-sheet simulation. If you are already familiar with the complexity range of GIA models, skip to Why FastIsostasy?. If you want to have a more thorough but still very accessbile introduction to GIA, we highly recommend reading Whitehouse et al. 2018. If you want to get started right away, feel free to directly go to the Examples. If you face any problem using the code or want to know more about the functionalities of the package, visit the API reference. If you face a problem you cannot solve, please open a GitHub issue with a minimal and reproduceable example.","category":"page"},{"location":"index.html#What-is-glacial-isostatic-adjustment?","page":"Introduction","title":"What is glacial isostatic adjustment?","text":"","category":"section"},{"location":"index.html","page":"Introduction","title":"Introduction","text":"The evolution of cryosphere components leads to changes in the ice and liquid water column and therefore in the vertical load applied upon the solid Earth. Glacial isostatic adjustment (GIA) denotes the mechanical response of the solid Earth, which is characterized by its vertical and horizontal displacement. GIA models usually encompass related processes, such as the resulting changes in sea-surface height and sea level.","category":"page"},{"location":"index.html","page":"Introduction","title":"Introduction","text":"The magnitude and time scale of GIA depends on the applied load and on solid-Earth parameters, here assumed to be the density, the viscosity and the lithospheric thickness. These parameters display a radial and sometimes also a lateral variability, further jointly denoted by parameter \"heterogeneity\". For further details, please refer to Wiens et al. 2021 and Ivins et al. 2023.","category":"page"},{"location":"index.html#Why-should-we-care?","page":"Introduction","title":"Why should we care?","text":"","category":"section"},{"location":"index.html","page":"Introduction","title":"Introduction","text":"GIA is known to present many feedbacks on ice-sheet evolution. Their net effect is negative, meaning that GIA inhibits ice-sheet growth and retreat. In other words, it tends to stabilize a given state and is therefore particularly important in the context of paleo-climate and climate change.","category":"page"},{"location":"index.html","page":"Introduction","title":"Introduction","text":"The speed and magnitude of anthropogenic warming is a potential threat to the Greenland and the West-Antarctic ice sheets. They both represent an ice volume that could lead to multi-meter sea-level rise. The effect of GIA in this context appears to be particularly relevant - not only from a theoretical but also from a practical perspective, as a large portion of human livelihoods are concentrated along coasts.","category":"page"},{"location":"index.html#Motivation","page":"Introduction","title":"Motivation","text":"","category":"section"},{"location":"index.html#Overview-of-GIA-for-ice-sheet-simulation","page":"Introduction","title":"Overview of GIA for ice-sheet simulation","text":"","category":"section"},{"location":"index.html","page":"Introduction","title":"Introduction","text":"GIA models present a wide range of complexity, which can only be briefly mentioned here. On the lower end, models such as the Elastic-Lithopshere/Viscous-Asthenopshere are (1) cheap to run and (2) easy to implement, which has made them popular within the ice-sheet modelling community. They present some acceptable limitations such as (3) regionally approximating a global problem and (4) lacking the radially layered structure of the solid Earth. However, some limitations have shown to be too important to be overlooked – mainly the fact that (5) the heterogeneity of the lithospheric thickness and upper-mantle viscosity cannot be represented.","category":"page"},{"location":"index.html","page":"Introduction","title":"Introduction","text":"On the higher end of the complexity spectrum, we find the 3D GIA models which address all the limitations of low-complexity models but are (1) expensive to run, (2) more tedious to couple to an ice-sheet model and (3) generally lack a well-documented and open-source code base. Due to these drawbacks, they do not represent a standard tool within the ice-sheet modelling community. Nonetheless, they are becoming increasingly used, as for instance in Gomez et al. 2018 and Van Calcar et al. 2023.","category":"page"},{"location":"index.html","page":"Introduction","title":"Introduction","text":"We here willingly omit to speak about 1D GIA models, as they lack the representation of heterogeneous solid-Earth parameters.","category":"page"},{"location":"index.html#Where-is-FastIsosatsy.jl-on-the-complexity-range?","page":"Introduction","title":"Where is FastIsosatsy.jl on the complexity range?","text":"","category":"section"},{"location":"index.html","page":"Introduction","title":"Introduction","text":"Although they are increasingly being coupled to ice-sheet models, we believe that the expense of 3D GIA models can be avoided while still addressing the aforementioned limitations of simplistic models. Models specifically designed for ice-sheet modelling, such as Bueler et al. 2007 and Coulon et al. 2021, have shown first improvements in closing the gap between simplistic and expensive models. FastIsostasy continues this work by generalizing both of these contributions into one, while benchmarking results against 1D and 3D GIA models.","category":"page"},{"location":"index.html","page":"Introduction","title":"Introduction","text":"FastIsostasy heavily relies on the Fast-Fourier Transform (FFT), as (1) its central PDE is solved by applying a Fourier collocation scheme and (2) important diagnostic fields are computed by matrix convolutions which can famously be accelerated by the use of FFT. FFT therefore inspired the name \"FastIsostasy\", along with a GitHub repository that eased the first steps of this package. The use of a performant language such as julia, as well as supporting performance-relevant computations on GPU allows FastIsostasy to live up to the expectations of low computation time.","category":"page"},{"location":"index.html","page":"Introduction","title":"Introduction","text":"We believe that FastIsostasy drastically reduces the burdens associated with using a 3D GIA model while offering all the complexity needed for ice-sheet modelling. As targeted and efficient climate-change mitigation relies on a good representation of important mechanisms in numerical models, we believe that this can be a significant contribution for future research.","category":"page"},{"location":"index.html#Technical-details","page":"Introduction","title":"Technical details","text":"","category":"section"},{"location":"index.html","page":"Introduction","title":"Introduction","text":"In case you wonder, FastIsostasy.jl:","category":"page"},{"location":"index.html","page":"Introduction","title":"Introduction","text":"Takes all parameters in SI units. This might be made more flexible in future by the use of Unitful.jl.\nRelies on a regular, square grid as those typically used for finite-difference schemes.\nHas a hybrid approach to solving its underlying PDE: while some terms are evaluated by finite differences, the usual expense of such method is avoided by applying a Fourier collocation scheme.\nFor now only supports square domains with the number of points being a power of 2. This can accelerate computations of FFTs but will be made more flexible in future work.","category":"page"},{"location":"index.html","page":"Introduction","title":"Introduction","text":"FastIsostasy.jl largely relies on following packages:","category":"page"},{"location":"index.html","page":"Introduction","title":"Introduction","text":"FFTW.jl\nInterpolations.jl\nCUDA.jl\nDSP.jl\nKalmanEnsembleProcesses.jl","category":"page"},{"location":"index.html#References","page":"Introduction","title":"References","text":"","category":"section"},{"location":"index.html","page":"Introduction","title":"Introduction","text":"Whitehouse et al. 2018\nWiens et al. 2021\nIvins et al. 2023.\nGomez et al. 2018\nVan Calcar et al. 2023\nBueler et al. 2007\nCoulon et al. 2021","category":"page"}]
}
