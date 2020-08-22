n = 1000                             # 数据个数
alpha <- rnorm(n, mean = 5, sd = 3)  # 生成alpha分布的数据
for(i in 1:n) while(alpha[i]<0) alpha[i]<-rnorm(1, mean = 5, sd = 3)  # 使得alpha > 0
x <- rbeta(n, alpha, 4)              # 生成beta分布的数据x
y <- sin(x)                          
y_avg <- mean(y)
y_avg


library("rstan")                     # 加载rstan包
data <- list(N = n, X = x)           # 准备mean.stan的数据
fit <- stan(file = 'mean.stan', data = data)
print(fit)
plot(fit)
