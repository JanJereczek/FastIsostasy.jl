
a = 6.371e6


θ = [0.0001, 0.001, 0.01, 0.02, 0.03, 0.04, 0.06, 0.08, 0.1, 0.16, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.8,
     1.0, 1.2, 1.6, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 12.0, 16.0, 20.0, 25.0, 30.0,
     40.0, 50.0, 60.0, 70.0, 80.0, 90.0]

function load_green_coeffs()
    # Column 1 (converted by some factor) of table A3 of
    # Deformation of the Earth by surface Loads, Farrell 1972
    rm=[ 0.0,    0.011,  0.111,  1.112,  2.224,  3.336,  4.448,  6.672,  8.896,  11.12, 17.79,
    22.24,  27.80,  33.36,  44.48,  55.60,  66.72,  88.96,  111.2,  133.4, 177.9,
    222.4,  278.0,  333.6,  444.8,  556.0,  667.2,  778.4,  889.6, 1001.0, 1112.0,
    1334.0, 1779.0, 2224.0, 2780.0, 3336.0, 4448.0, 5560.0, 6672.0, 7784.0, 8896.0,
    10008.0] .* 1e3
    # converted to meters
    # GE /(10^12 rm) is vertical displacement in meters (applied load is 1kg)

    # Column 2 of table A3 of
    # Deformation of the Earth by surface Loads, Farrell 1972
    GE=[-33.6488, -33.64, -33.56, -32.75, -31.86, -30.98, -30.12, -28.44, -26.87, -25.41,
        -21.80, -20.02, -18.36, -17.18, -15.71, -14.91, -14.41, -13.69, -13.01,
        -12.31, -10.95, -9.757, -8.519, -7.533, -6.131, -5.237, -4.660, -4.272,
        -3.999, -3.798, -3.640, -3.392, -2.999, -2.619, -2.103, -1.530, -0.292,
        0.848,  1.676,  2.083,  2.057,  1.643];
    return rm, GE
end

function get_II()
    # dimensions of load element
    dx=2*Lx*1000/Nx
    dy=2*Ly*1000/Ny
    
    II=zeros(2*Nx-1,2*Ny-1);

    # compute entries of II by dblquad, using quad method and default TOL=1e-6 
    for p=-Nx+1:Nx-1
        for q=-Ny+1:Ny-1
            II[p+Nx,q+Ny] = dblquad( @integrand, -dx/2, dx/2, -dy/2, dy/2 );
        end
    end
    return II
end

function integrand(xi, eta)
    r = sqrt( (p*dx-xi).^2 + (q*dy-eta).^2 )
    z = zeros(  )

end

function Quad2D(f, n, x1, x2, y1, y2)
    x, w = gausslegendre( n );
    mx, px = get_lin_transform_2_norm(x1, x2)
    my, py = get_lin_transform_2_norm(y1, y2)
    sum = 0
    for i=1:n, j=1:n
        sum = sum + f(
            lin_transform_2_norm(x[i], mx, px),
            lin_transform_2_norm(x[j], my, py),
        ) * w[i] * w[j] / mx / my
    end
    return sum
end

function Quad1D(f, n, x1, x2)
    x, w = gausslegendre( n )
    m, p = get_lin_transform_2_norm(x1, x2)
    sum = 0
    for i=1:n
        sum = sum + f(lin_transform_2_norm(x[i], m, p)) * w[i] / m
    end
    return sum
end

function get_lin_transform_2_norm(x1, x2)
    x1_norm, x2_norm = -1, 1
    m = (x2_norm - x1_norm) / (x2 - x1)
    p = x1_norm - m * x1
    return m, p
end

function lin_transform_2_norm(y, m, p)
    return (y-p)/m
end