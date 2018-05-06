import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import os
import re
import numpy as np
import seaborn
import tensorflow as tf
import datetime
import time

from sklearn.preprocessing import scale
from sklearn.decomposition import PCA
from sklearn.metrics import recall_score, precision_score

from mpl_toolkits.mplot3d import Axes3D

def draw_2d_embeddings(model, labels):

	fig = plt.figure()

	x_0 = model.graph_embeddings[np.where(labels == 0)]
	x_1 = model.graph_embeddings[np.where(labels == 1)]

	plt.scatter(x_0[:, 0], x_0[:, 1], c='b')
	plt.scatter(x_1[:, 0], x_1[:, 1], c='r')

	plt.title("2D embeddings")
	plt.xlabel('1 feature')
	plt.ylabel('2 feature')

	plt.grid()
	plt.show()

def draw_3d_embeddings(model, labels):

	fig = plt.figure()

	ax = fig.add_subplot(111, projection='3d')

	x_0 = model2.graph_embeddings[np.where(labels == 0)]
	x_1 = model2.graph_embeddings[np.where(labels == 1)]

	ax.scatter(x_0[:, 0], x_0[:, 1], x_0[:, 2], c='b')
	ax.scatter(x_1[:, 0], x_1[:, 1], x_1[:, 2], c='r')

	plt.show()

def iterative_drawing_2d(model, labels):

	fig = plt.figure()
	ax = fig.add_subplot(111)
	plt.ion()

	fig.show()
	fig.canvas.draw()

	for i, label in zip(range(len(labels)), labels):
	    if label == 0:
	        ax.scatter(model.graph_embeddings[i][0], model.graph_embeddings[i][1], c='b')
	    else:
	        ax.scatter(model.graph_embeddings[i][0], model.graph_embeddings[i][1], c='r')
	    fig.canvas.draw()
	    time.sleep(0.01)


def iterative_drawing_2d_array(arr, labels):

	fig = plt.figure()
	ax = fig.add_subplot(111)
	plt.ion()

	fig.show()
	fig.canvas.draw()

	for i, label in zip(range(len(labels)), labels):
	    if label == 0:
	        ax.scatter(array[i][0], array[i][1], c='b')
	    else:
	        ax.scatter(array[i][0], array[i][1], c='r')
	    fig.canvas.draw()
	    time.sleep(0.0)
