module SpectralUtils

const VecI = AbstractVector
const VALN = Val('N')
const VALT = Val('T')
const IFTHREADS = isone(Threads.nthreads()) ? Val('N') : Val('T')

include("./1dfft.jl")

end # module SpectralUtils

