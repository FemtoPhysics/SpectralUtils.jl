## power of radix 2 (flooring mode)
function pwr2(x::Int)
    x ≡ 0 && error("pwr2(x): x cannot be zero!")
    r = 0
    x > 4294967295 && (x >>= 32; r += 32)
    x >      65535 && (x >>= 16; r += 16)
    x >        255 && (x >>=  8; r +=  8)
    x >         15 && (x >>=  4; r +=  4)
    x >          3 && (x >>=  2; r +=  2)
    x >          1 && (x >>=  1; r +=  1)

    return r
end

function twiddle!(wa::VecI{Complex{T}}) where T<:AbstractFloat
    Nby2 = length(wa)
    if isone(Nby2)
        @inbounds wa[1] = complex(1.0, -0.0)

        return wa
    end

    Nby4 = Nby2 >> 1
    if isone(Nby4)
        @inbounds wa[1] = complex(1.0, -0.0)
        @inbounds wa[2] = complex(0.0, -1.0)

        return wa
    end

    Nby8 = Nby4 >> 1
    if isone(Nby8)
        tmp = -0.7071067811865476
        @inbounds wa[1] = complex( 1.0, -0.0)
        @inbounds wa[2] = complex(-tmp,  tmp)
        @inbounds wa[3] = complex( 0.0, -1.0)
        @inbounds wa[4] = complex( tmp,  tmp)

        return wa
    end

    invNby2 = inv(Nby2)   # 2 / N
    cosθ = cospi(invNby2) # cos(2π / N)
    sinθ = sinpi(invNby2) # sin(2π / N)

    let tmp = -0.7071067811865476
        @inbounds wa[1]            = complex( 1.0, -0.0)
        @inbounds wa[Nby8 + 1]     = complex(-tmp,  tmp)
        @inbounds wa[Nby4 + 1]     = complex( 0.0, -1.0)
        @inbounds wa[3 * Nby8 + 1] = complex( tmp,  tmp)
    end

    jj = Nby4
    kk = Nby4 + 2
    ll = Nby2

    for ii in 2:Nby8
        prevReal, prevImag = @inbounds reim(wa[ii - 1])
        nextReal = prevReal * cosθ + prevImag * sinθ
        nextImag = prevImag * cosθ - prevReal * sinθ
        @inbounds wa[ii] = complex( nextReal,  nextImag)
        @inbounds wa[jj] = complex(-nextImag, -nextReal)
        @inbounds wa[kk] = complex( nextImag, -nextReal)
        @inbounds wa[ll] = complex(-nextReal,  nextImag)
        jj -= 1
        kk += 1
        ll -= 1
    end

    return wa
end

#=
Cooley-Tukey butterfly computation:
-----------------------------------
    1. ya := destination array
    2. xa := source array
    3. wa := twiddle factors array
    4. sᵢ := starting index
    5. hs := half of FFT size
    6. ns := number of steps
    7. ss := step size of the current subproblem
    8. pd := index difference of butterfly computation pair
=#
function butterfly!(ya::VecI{Complex{T}}, xa::VecI{Complex{T}}, wa::VecI{Complex{T}},
                    sᵢ::Int, hs::Int, ns::Int, ss::Int, pd::Int) where T<:AbstractFloat
    wᵢ = 1
    yᵢ = xᵢ = sᵢ
    for _ in 1:ns
        xⱼ = xᵢ + hs
        
        @inbounds ya[yᵢ]    = (xa[xᵢ] + xa[xⱼ])
        @inbounds ya[yᵢ+pd] = (xa[xᵢ] - xa[xⱼ]) * wa[wᵢ]
        
        yᵢ += ss
        xᵢ += pd
        wᵢ += pd
    end

    return nothing
end

#=
Decimation-in-time FFT with a naturally ordered input-output:
-------------------------------------------------------------
    1. sa := signal array
    2. ba := buffer array
    3. wa := twiddle factors array
    4. sf := switch flag
=#
function ditnn!(::Val{'N'}, sa::VecI{Complex{T}}, ba::VecI{Complex{T}}, wa::VecI{Complex{T}}, hs::Int) where T<:AbstractFloat
    ns = hs
    pd = 1
    ss = 2
    sf = false

    while ns > 0
        if sf
            for sᵢ in 1:pd
                butterfly!(sa, ba, wa, sᵢ, hs, ns, ss, pd)
            end
        else
            for sᵢ in 1:pd
                butterfly!(ba, sa, wa, sᵢ, hs, ns, ss, pd)
            end
        end

        ns >>= 1
        pd <<= 1
        ss <<= 1
        sf = !sf
    end

    return nothing
end

# Multithreading has its benefit when the FFT size (`hs`) ≥ 8192.
function ditnn!(::Val{'T'}, sa::VecI{Complex{T}}, ba::VecI{Complex{T}}, wa::VecI{Complex{T}}, hs::Int) where T<:AbstractFloat
    ns = hs
    pd = 1
    ss = 2
    sf = false

    while ns > 0
        if sf
            Threads.@threads for sᵢ in 1:pd
                butterfly!(sa, ba, wa, sᵢ, hs, ns, ss, pd)
            end
        else
            Threads.@threads for sᵢ in 1:pd
                butterfly!(ba, sa, wa, sᵢ, hs, ns, ss, pd)
            end
        end

        ns >>= 1
        pd <<= 1
        ss <<= 1
        sf = !sf
    end

    return nothing
end
