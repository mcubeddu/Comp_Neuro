import csv
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

with open('im1.csv', 'r') as im1:
	img1 = []
	csv_reader = csv.reader(im1, delimiter=',')
	for row in csv_reader:
		img1.append(list(map(lambda rgb: float(rgb), row)))

with open('im2.csv', 'r') as im2:
	img2 = []
	csv_reader = csv.reader(im2, delimiter=',')
	for row in csv_reader:
		img2.append(list(map(lambda rgb: float(rgb), row)))

with open('im3.csv', 'r') as im3:
	img3 = []
	csv_reader = csv.reader(im3, delimiter=',')
	for row in csv_reader:
		img3.append(list(map(lambda rgb: float(rgb), row)))

img1 = np.array(img1)
img2 = np.array(img2)
img3 = np.array(img3)
std_shape = img1.shape
X = np.vstack([img1.flatten(), img2.flatten(), img3.flatten()])
print(f'[MEAN] = {X.mean(1)}\n')
print(f'[Variance]:\n\n{np.var(X, axis=1)}\n\n')
print(f'[Skewness]:\n\n{stats.skew(X, axis=1)}\n\n')
print(f'[Kurtosis]:\n\n{stats.kurtosis(X, axis=1)}\n\n')

# Part 2
centered = X.T - np.mean(X, axis=1)
C = np.cov(centered.T)
eig_vals, eig_vecs = np.linalg.eig(C)
decorrelated = X.T.dot(eig_vecs)
# avoid dividing by 0
whitened = decorrelated / np.sqrt(eig_vals + 1e-10)

# Part 3

def kurt_grad(w1, x_tilde):
	wT_x = w1.dot(x_tilde)
	p = len(x_tilde.T)
	A = np.sum(x_tilde.T.dot(w1))
	A_2 = np.sum(x_tilde.T.dot(w1)**2)
	A_3 = np.sum(x_tilde.T.dot(w1)**3)
	A_4 = np.sum(x_tilde.T.dot(w1)**4) 
	D = 4*p*((A_3/(A_2**2)) - (A*A_4/(A_2**3)))
	x_mu = np.mean(x_tilde, axis=0)
	d_w1 = D*x_mu[0]/p
	d_w2 = D*x_mu[1]/p
	d_w3 = D*x_mu[2]/p
	res = np.array([d_w1, d_w2, d_w3])
	return res

l_rate = 1e-7
Wt = []
# FIND w_1
# w_next = np.random.normal(0,0.2,3)
w_next = np.array([-1e-6,1e-6,-1e-6])
w_curr = np.array([.9,.9,.9])
max_iter = 1e3
ctr = 0
while np.linalg.norm(w_next - w_curr) > 1e-5:
	w_curr = w_next
	ctr += 1
	if ctr % 100 == 0:
		print(w_curr)
	if ctr >= max_iter:
		print('max iter reached')
		break
	w_next = w_curr + l_rate*kurt_grad(w_curr, whitened.T)
	w_next = w_next/np.linalg.norm(w_next)
w_1 = w_next
Wt.append(w_1)

# FIND w_2
# w_next = np.random.normal(0,0.2,3)
w_next = np.array([1e-6,1e-6,-1e-6])
w_curr = np.array([.9,.9,.9])
max_iter = 1e3
ctr = 0
while np.linalg.norm(w_next - w_curr) > 1e-5:
	w_curr = w_next
	ctr += 1
	if ctr >= max_iter:
		print('max iter reached')
		break
	w_next = w_curr + l_rate*kurt_grad(w_curr, whitened.T)
	w1_proj = w_next.dot(w_1)
	w_next = (w_next - w_1*w1_proj)/np.linalg.norm(w_next)
w_2 = w_next
Wt.append(w_2)

# FIND w_3
# w_next = np.random.normal(0,1e-4,3)
w_next = np.array([-1e-6,1e-6,-1e-6])
w_curr = np.array([.9,.9,.9])
max_iter = 1e3
ctr = 0
while np.linalg.norm(w_next - w_curr) > 1e-5:
	w_curr = w_next
	ctr += 1
	if ctr >= max_iter:
		print('max iter reached')
		break
	w_next = w_curr + l_rate*kurt_grad(w_curr, whitened.T)
	w1_proj = w_next.dot(w_1)
	w2_proj = w_next.dot(w_2)
	w_next = (w_next - w_1*w1_proj - w_2*w2_proj)/np.linalg.norm(w_next)
w_3 = w_next
Wt.append(w_3)
print('W = ')
for col in Wt: print(col)
plt.figure()
plt.axis('off')
plt.subplot(3,4,1)
plt.imshow(img1, cmap='Greys')
plt.title('Mixed 1')
plt.subplot(3,4,5)
plt.imshow(img2, cmap='Greys')
plt.title('Mixed 2')
plt.subplot(3,4,9)
plt.imshow(img3, cmap='Greys')
plt.title('Mixed 3')
plt.subplot(3,4,2)
plt.imshow(whitened.T[0].reshape(std_shape), cmap='Greys')
plt.title('Whitened 1')
plt.subplot(3,4,6)
plt.imshow(whitened.T[1].reshape(std_shape), cmap='Greys')
plt.title('Whitened 2')
plt.subplot(3,4,10)
plt.imshow(whitened.T[2].reshape(std_shape), cmap='Greys')
plt.title('Whitened 3')
plt.subplot(3,4,3)
plt.imshow(whitened.dot(w_1).reshape(std_shape), cmap='Greys')
plt.title('Recovered 1(+)')
plt.subplot(3,4,7)
plt.imshow(whitened.dot(w_2).reshape(std_shape), cmap='Greys')
plt.title('Recovered 2(+)')
plt.subplot(3,4,11)
plt.imshow(whitened.dot(w_3).reshape(std_shape), cmap='Greys')
plt.title('Recovered 3(+)')
plt.subplot(3,4, 4)
plt.imshow(whitened.dot(-w_1).reshape(std_shape), cmap='Greys')
plt.title('Recovered 1(-)')
plt.subplot(3,4,8)
plt.imshow(whitened.dot(-w_2).reshape(std_shape), cmap='Greys')
plt.title('Recovered 2(-)')
plt.subplot(3,4,12)
plt.imshow(whitened.dot(-w_3).reshape(std_shape), cmap='Greys')
plt.title('Recovered 3(-)')
plt.savefig('images.png', dpi=500)