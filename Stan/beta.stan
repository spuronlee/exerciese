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
  real<lower=0> sinX_alpha;
  real<lower=0> sinX_beta;
}


model {            // Model block.
  
  // priors
  alpha ~ normal(5, 3);
  X ~ beta(alpha, 4); 
  
  // likelihood
  sinX ~ beta(sinX_alpha, sinX_beta);
}

