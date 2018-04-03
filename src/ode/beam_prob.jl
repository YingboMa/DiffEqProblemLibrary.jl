"""
[Beam Problem](https://archimede.dm.uniba.it/~testset/report/beam.pdf) (Stiff)

It is in the form of ``z''=f(t,z,z'), \\quad z(0)=z_0 z'(0)=z'_0,`` with

```math
y \\in ℝ^n, \\quad 0 ≤ t

where ``f`` is defined by

```math
f(t, z, z') = Cv + Du
```

Here ``C`` is the tridiagonal ``n × n`` matrix whose entries are given by

```math
\\begin{cases}
C_{1,1} = 1&, C_{n,n} = 3, and C_{l,l} = 2, \\\\
C_{l,l+1} &= -\\cos(z_l - z_{l+1}), \\\\
C_{l,l-1} &= -\\cos(z_l - z_{l-1}),
\\end{cases}
```

and D is the ``n × n`` bidiagonal matrix whose lower and upper diagonal entries are

```math
\\begin{cases}
D_{l,l+1} &= -\\sin(z_l - z_{l+1}), \\\\
D_{l,l-1} &= -\\sin(z_l - z_{l-1}).
\\end{cases}
```

`v=(v_1, v_2, \\cdots, v_n)^T` is defined by

```math
v_l = n^4(z_{l-1} - 2z_l + z_{l+1}) + n^2(\\cos(z_l)F_y - \\sin(z_l)F_x), \\quad l = 1,\\cdots, n
```

with ``z_0 = -z_1, z_{n+1} = z_{n},`` and ``u`` is the column vector of size ``n`` solution of the tridiagonal system

```math
Cu = g
```

with ``g=Dv + (z'^2_1, z'^2_2, \\cdots, z'^2_n)^T``.
"""
function calc_v!(v, z, fx, fy)
  N = length(v)
  for i in 2:N-1
    v[i] = N^4*(z[i-1]-2z[i]+z[i+1]) + N^2*(cos(z[i])*fy - sin(z[i])*fx)
  end
  i = 1
  v[i] = N^4*(-z[i]-2z[i]+z[i+1]) + N^2*(cos(z[i])*fy - sin(z[i])*fx)
  i = N
  v[i] = N^4*(z[i-1]-2z[i]+z[i]) + N^2*(cos(z[i])*fy - sin(z[i])*fx)
  v
end
function calc_C!(C, z)
  N = length(z)
  C[1,1] = 1
  C[N,N] = 3
  for i in 1:N
    C[i,i] = 2
  end
  for i in 1:N-1
    C[i, i+1] = -cos(z[i]-z[i+1])
    i = i+1
    C[i, i-1] = -cos(z[i]-z[i-1])
  end
  C
end
function calc_D!(D, z)
  N = length(z)
  for i in 1:N-1
    D[i, i+1] = -sin(z[i]-z[i+1])
    i = i+1
    D[i, i-1] = -sin(z[i]-z[i-1])
  end
  D
end
function calc_g(g, D, dz, v)
  A_mul_B!(g, D, v)
  @. g = g + dz
end
function beam_fun(N)
  v = zeros(N)
  u = zeros(N)
  C = Tridiagonal(zeros(N), zeros(N), zeros(N))
  D = Tridiagonal(zeros(N), zeros(N), zeros(N))
  beam = (ddz, dz, z, p, t) -> begin
    f = p[1]
    fx, fy = f(t)
    calc_v!(v, z, fx, fy)
    calc_C!(C, z)
    calc_D!(D, z)
    # alias
    g = ddz
    calc_g!(g, D, dz, v)
    LUC = lufact!(C)
    A_ldiv_B!(u, C, g)
    A_mul_B!(ddz, C, v)
    BLAS.gemm('N', 'N', 1., D, u, 1., ddz)
  end
  beam
end
