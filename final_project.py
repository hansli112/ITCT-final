import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import sys
import threading, queue
from pyentrp import entropy as ent
import lzma

def entropy(X):
	uni_array, counts = np.unique(X, return_counts=True)
	norm_counts = counts / counts.sum()
	return -(norm_counts * np.log(norm_counts)).sum()

def statistical_complexity(X): # without nomalization
	def JSD(X, Y):
		return entropy((X + Y) / 2) - (entropy(X) + entropy(Y)) / 2

	d = X.shape[0]
	#d_fact = np.math.factorial(d)
	#kappa = -((1 + (1 / d_fact)) * np.log(d_fact + 1) + np.log(d_fact) - 2 * np.log(2 * d_fact)) / 2
	kappa = 1
	return (1 / kappa) * JSD(X, np.full(d, 1 / d)) * ent.permutation_entropy(X)

def complexity_ratio(X):
	X_bytes = X.tobytes()
	return len(lzma.compress(X_bytes)) / len(X_bytes)

def get_features(song, q, lst):
	TS_T, sr = librosa.load(song)
	TS_S = np.fft.fft(TS_T)

	H_T = entropy(TS_T)
	PE_T = ent.permutation_entropy(TS_T)
	C_T = statistical_complexity(TS_T)
	CR_T = complexity_ratio(TS_T)
	H_S = entropy(TS_S)
	PE_S = ent.permutation_entropy(TS_S)
	C_S = statistical_complexity(TS_S)
	CR_S = complexity_ratio(TS_S)

	feature = np.array((H_T, PE_T, C_T, CR_T, H_S, PE_S, C_S, CR_S))

	q.put(feature)
	lst.append(song.split('/')[-1].split('.')[0])
	print("finished %s" % song)

def MDS(X):
	D = np.arccos(np.matmul(X, X.transpose()).clip(-1, 1))
	J = np.eye(D.shape[0]) - (np.full(D.shape, 1) / D.shape[0])
	B = -np.matmul(np.matmul(J, D), J) / 2
	eig_l, eig_v = np.linalg.eigh(B)
	return np.flip(np.matmul(eig_v[:, -2:], np.diag(np.sqrt(eig_l[-2:]))), axis=1)

threads = []
q = queue.Queue()
name = []
songs = np.empty((0, 8))
with open(sys.argv[1], 'r') as song_list:
	for song in song_list:
		song = song.strip()

		thread = threading.Thread(target=get_features, args=(song, q, name))
		thread.start()
		threads.append(thread)

for thread in threads:
	thread.join()

while not q.empty():
	songs = np.vstack((songs, q.get()))

# normalize
songs_mean = songs - songs.mean(axis=0)
songs = songs_mean / np.linalg.norm(songs_mean, axis=0)
songs_mean = songs - songs.mean(axis=1).reshape((-1, 1))
songs = songs_mean / np.linalg.norm(songs_mean, axis=1).reshape((-1, 1))

# MDS algorithm
result = MDS(songs)
plt.plot(result[:, 0], result[:, 1], '.-')
for i in range(songs.shape[0]):
	plt.annotate(name[i], (result[i, 0], result[i, 1]), xytext=(0, 3), textcoords="offset pixels", ha='center', va='bottom')
plt.savefig("figure.png")
plt.show()
