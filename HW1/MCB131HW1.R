f = function(x, sigma, w_0){
	t1 = exp(-(x^2)/(2*sigma^2))
	t2 = cos(w_0 * x)
	return(t1*t2)
}

# Fourier transform of f(x)
fhat = function(omega, sigma, w_0){
c = sigma/2
t1 = exp(-(sigma^2)*((omega-w_0)^2)/2)
t2 = exp(-(sigma^2)*((omega+w_0)^2)/2)
return(c*(t1+t2))
}
sig = 1
omegas = seq(-3, 3, by=0.02)
xs = seq(-6, 6, by = 0.05)

for (w.0 in c(0.1, 1.0, 10.0)){
# {
# par(mfrow=c(1,2), fig = c(0, 1, 0.5, 0.5))
fs = f(xs, sig, w.0)
fhats = fhat(omegas, sig, w.0)
# plot(xs, fs, 'l', xlab = 'x', ylab = 'f(x)', main = sprintf('f(x) for w_0 = %f', w.0))
# plot(omegas, fhats, 'l', xlab='w', ylab = 'F(w)', main = sprintf('F(w) for w_0 = %f', w.0))
# }
# make labels and margins smaller
par(cex=0.7, mai=c(0.05,0.1,0.2,0.1))
# define area for the histogram
par(fig=c(0.05,0.45,0.3,0.9))
plot(xs, fs, 'l', xlab = 'x', ylab = 'f(x)', main = sprintf('f(x) for w_0 = %f', w.0))
# define area for the boxplot
par(fig=c(0.5,0.95,0.3,0.9), new=TRUE)
plot(omegas, fhats, 'l', xlab='w', ylab = 'F(w)', main = sprintf('F(w) for w_0 = %f', w.0))
}
