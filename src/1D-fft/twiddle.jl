function twiddle!(warr::VecI{Complex{T}}) where T<:Real
    Nby2 = length(warr)
    if isone(Nby2)
        @inbounds warr[1] = complex(1.0, -0.0)
        return warr
    end

    Nby4 = Nby2 >> 1
    if isone(Nby4)
        @inbounds warr[1] = complex(1.0, -0.0)
        @inbounds warr[2] = complex(0.0, -1.0)
        return warr
    end

    Nby8 = Nby4 >> 1
    if isone(Nby8)
        tmp = -0.7071067811865476
        @inbounds warr[1] = complex( 1.0, -0.0)
        @inbounds warr[2] = complex(-tmp,  tmp)
        @inbounds warr[3] = complex( 0.0, -1.0)
        @inbounds warr[4] = complex( tmp,  tmp)
        return warr
    end

    invNby2 = inv(Nby2) # 2π / N
    cosθ = cospi(invNby2) # cos(2π / N)
    sinθ = sinpi(invNby2) # sin(2π / N)

    let tmp = -0.7071067811865476
        @inbounds warr[1]            = complex( 1.0, -0.0)
        @inbounds warr[Nby8 + 1]     = complex(-tmp,  tmp)
        @inbounds warr[Nby4 + 1]     = complex( 0.0, -1.0)
        @inbounds warr[3 * Nby8 + 1] = complex( tmp,  tmp)
    end

    jj = Nby4
    kk = Nby4 + 2
    ll = Nby2

    for ii in 2:Nby8
        prevReal, prevImag = @inbounds reim(warr[ii - 1])
        nextReal = prevReal * cosθ + prevImag * sinθ
        nextImag = prevImag * cosθ - prevReal * sinθ
        @inbounds warr[ii] = complex( nextReal,  nextImag)
        @inbounds warr[jj] = complex(-nextImag, -nextReal)
        @inbounds warr[kk] = complex( nextImag, -nextReal)
        @inbounds warr[ll] = complex(-nextReal,  nextImag)
        jj -= 1
        kk += 1
        ll -= 1
    end

    return warr
end
