# 生成数据
n = 50000
alpha <- rnorm(n, mean = 5, sd = 3)
for(i in 1:n) while(alpha[i]<0) alpha[i]<-rnorm(1, mean = 5, sd = 3)
x <- rbeta(n, alpha, 4)
y <- sin(x)

# 密度曲线
z=density(y, na.rm=T) #na.rm=T 忽略缺省值
plot(z, main = "Distribution of Sample")

# 估计y=sin(x)属于哪种分布
library(fitdistrplus)
descdist(y)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             

fitdist(y, "norm")
fitdist(y, "beta",method = "mme")
fitdist(y, "beta",method = "mge")


library("rstan")          
data <- list(N = n, X = x) 


# 假设y=sin(x)属于正态分布
fit <- stan(file = 'norm.stan', data = data)
print(fit)
plot(fit)


# 假设y=sin(x)属于贝塔分布
fit <- stan(file = 'beta.stan', data = data)
print(fit)
plot(fit)
