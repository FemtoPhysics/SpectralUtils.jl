include("./radix2.jl")

struct FFT{S<:AbstractFloat}
    cache  ::Vector{Complex{Float64}}
    twiddle::Vector{Complex{Float64}}
    fftsize::Int
    ifSwap ::Bool

    function FFT(fftsize::Int) # type-stability ✓
        # fftsize should be power of 2
        cache   = Vector{Complex{Float64}}(undef, fftsize)
        twiddle = Vector{Complex{Float64}}(undef, fftsize >> 1)

        return new{Float64}(
            cache, twiddle!(twiddle), fftsize, isone(pwr2(fftsize) & 1)
        )
    end
end

function fft!(x::VecI{Complex{T}}, f::FFT{T}) where T<:AbstractFloat
    if f.ifSwap
        ditnn!(VALN, copyto!(f.cache, x), x, f.twiddle, f.fftsize >> 1)
    else
        ditnn!(VALN, x, f.cache, f.twiddle, f.fftsize >> 1)
    end

    return x
end

function ifft!(x::VecI{Complex{T}}, f::FFT{T}) where T<:AbstractFloat
    fftsize = f.fftsize
    cache = f.cache

    if f.ifSwap
        @simd for i in eachindex(x)
            @inbounds cache[i] = conj(x[i])
        end
        ditnn!(VALN, cache, x, f.twiddle, fftsize >> 1)
    else
        @simd for i in eachindex(x)
            @inbounds x[i] = conj(x[i])
        end
        ditnn!(VALN, x, cache, f.twiddle, fftsize >> 1)
    end

    @simd for i in eachindex(x)
        @inbounds x[i] = conj(x[i]) / fftsize
    end

    return x
end
