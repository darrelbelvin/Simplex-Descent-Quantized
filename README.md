# Nelder-Mead-Belvin
A quantized implementation of a simplex descent algorithm inspired by the Nelder-Mead algorithm

Quantized means that this algorithm will only pass multiples of a quanta as arguments to the cost function. It could be viewed as a grid search optimized with a Nelder-Mead like algorithm

As a result of being quantized, basin mapping becomes feasable and finding all local maxima in a grid and thus also finding the global maximum is possible. If basin mapping is turned on, the entire grid will be represented in memory as an int for each grid square, so please keep memory requirements in mind if using basin mapping.