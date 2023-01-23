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
                    sᵢ::Int, hs::Int, ns::Int, ss::Int, pd::Int) where T<:Real
    wᵢ = 1
    yᵢ = xᵢ = sᵢ
    for _ in 1:ns
        yⱼ = yᵢ + pd
        
        @inbounds Xᵢ = xa[xᵢ]
        @inbounds Xⱼ = xa[xᵢ + hs]
        @inbounds ya[yᵢ] = Xᵢ + Xⱼ
        @inbounds ya[yⱼ] = wa[wᵢ] * (Xᵢ - Xⱼ)
        
        yᵢ += ss
        xᵢ += pd
        wᵢ += pd
    end
end

#=
Decimation-in-time FFT with a naturally ordered input-output:
-------------------------------------------------------------
    1. sa := signal array
    2. ba := buffer array
    3. wa := twiddle factors array
    4. sf := switch flag
=#
function ditnn!(::Val{'N'}, sa::VecI{Complex{T}}, ba::VecI{Complex{T}}, wa::VecI{Complex{T}}, hs::Int) where T<:Real
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
function ditnn!(::Val{'T'}, sa::VecI{Complex{T}}, ba::VecI{Complex{T}}, wa::VecI{Complex{T}}, hs::Int) where T<:Real
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
