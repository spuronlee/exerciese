# ��������
n = 50000
alpha <- rnorm(n, mean = 5, sd = 3)
for(i in 1:n) while(alpha[i]<0) alpha[i]<-rnorm(1, mean = 5, sd = 3)
x <- rbeta(n, alpha, 4)
y <- sin(x)

# �ܶ�����
z=density(y, na.rm=T) #na.rm=T ����ȱʡֵ
plot(z, main = "Distribution of Sample")

# ����y=sin(x)�������ֲַ�
library(fitdistrplus)
descdist(y)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             

fitdist(y, "norm")
fitdist(y, "beta",method = "mme")
fitdist(y, "beta",method = "mge")


library("rstan")           # ����rstan��
data <- list(N = n, X = x) # ׼��mean.stan������


# ����y=sin(x)������̬�ֲ�
fit <- stan(file = 'norm.stan', data = data)
print(fit)
plot(fit)


# ����y=sin(x)���ڱ����ֲ�
fit <- stan(file = 'beta.stan', data = data)
print(fit)
plot(fit)