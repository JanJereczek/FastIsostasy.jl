# API reference

## Basic structs

```@docs
KernelMatrix
ComputationDomain
PhysicalConstants
LayeredEarth
RefGeoState
GeoState
FastIsoTools
FastIsoProblem
```

## Mechanics

```@docs
solve!(::FastIsoProblem)
init
step!
update_diagnostics!
dudt_isostasy!
update_elasticresponse!
```

## Parameter inversion

```@docs
InversionConfig
InversionData
InversionProblem
solve!(::InversionProblem)
```
## Utils

```@docs
years2seconds
seconds2years
m_per_sec2mm_per_yr
meshgrid
scalefactor
latlon2stereo
stereo2latlon
write_out!
```

[^Goelzer2020]:
    Heiko Goelzer et al. (2020): [Brief communication: On calculating the sea-level contribution in marine ice-sheet models](https://doi.org/10.5194/tc-14-833-2020)

[^Snyder1987]:
    John Snyder (1987): [Map projections -- A working manual](https://pubs.er.usgs.gov/publication/pp1395)

[^Farrell1972]:
    William Farrel (1972): [Deformation of the Earth by surface Loads, Farell 1972](https://doi.org/10.1029/RG010i003p00761)