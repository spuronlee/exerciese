//
// This Stan program defines a simple model, with a
// vector of values 'y' modeled as normally distributed
// with mean 'mu' and standard deviation 'sigma'.
//
// Learn more about model development with Stan at:
//
//    http://mc-stan.org/users/interfaces/rstan.html
//    https://github.com/stan-dev/rstan/wiki/RStan-Getting-Started
//

// The input data is a vector 'y' of length 'N'.
data {              // Data block.
  int<lower=0> N;   // number of data
  vector[N] X;      // data
}

transformed data {  // Transformed data block.
  vector[N] sinX;   
  sinX = sin(X);    // transforme x to sin(x)
} 

parameters {        // Parameters block.
  real<lower=0> alpha;
  real<lower=0> mu_X;
  real<lower=0> sigma_X;
}


model {            // Model block.
  
  // priors
  alpha ~ normal(5, 3);
  X ~ beta(alpha, 4); 
  
  // likelihood
  target += normal_lpdf(sinX | mu_X, sigma_X);
}
