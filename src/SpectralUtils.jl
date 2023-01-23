module SpectralUtils

const VecI = AbstractVector
const VALN = Val('N')
const VALT = Val('T')
const IFTHREADS = isone(Threads.nthreads()) ? Val('N') : Val('T')

include("./1D-fft/twiddle.jl")
include("./1D-fft/ditnn.jl")

end # module SpectralUtils

