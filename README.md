# Markov chain Monte Carlo (MCMC) on high-dimensional spheres

For my Master's dissertation, supervised by [Dr Tim Sullivan][2], I am investigating MCMC algorithms on high-dimensional spheres. This closely follows the paper [Dimension-independent Markov chain Monte Carlo on the sphere][1] by H.C. Lie, D. Rudolf, B. Sprungk and T.J. Sullivan.

This repo currently contains a variety of algorithms in Euclidean space, from symmetric random walk Metropolis-Hastings to preconditioned Crank-Nicolson (pCN); I am currently working on implementing other algorithms such as the Gibbs sampler, Metropolis-adjusted Langevin approximation (MALA) and Hamiltonian MCMC, as well as starting to look at moving these methods to the sphere.

[1]: <https://arxiv.org/pdf/2112.12185.pdf> "Dimension-independent Markov chain Monte Carlo on the sphere"
[2]: <https://warwick.ac.uk/fac/sci/maths/people/staff/sullivan/> "Dr Tim Sullivan"