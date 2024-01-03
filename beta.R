library(reticulate)
np <- import('numpy')

rm(list = ls())

#Beta mixture

x <- seq(0, 1, length.out = 100)
y1 <- dbeta(x, 2, 8)
y2 <- dbeta(x, 8, 2)
y <- (y1 + y2) / 2
plot(x, y1, type = 'l')
plot(x, y2, type = 'l')
plot(x, y, type = 'l')

num_samples <- 10000
samples <- rep(0, num_samples)

for (n in 1:num_samples) {
  mix <- rbinom(1, 1, 0.5)
  if (mix == 1){
    samples[n] <- rbeta(1, 2, 8)
  } else {
    samples[n] <- rbeta(1, 8, 2) 
  }
}

hist(samples, breaks = 30)

setwd("~/dev/mcmc_sphere")
np$save('./output/mixed_beta_samples.npy', samples)
