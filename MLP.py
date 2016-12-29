# coding: utf8
# !/usr/bin/env python
# ------------------------------------------------------------------------
# Implémentation d'un MLP avec rétropropagation
# Écrit par Stanislas LEROY
#
# Distribué sous licence BSD.
# ------------------------------------------------------------------------

import gzip # pour décompresser les données
import cPickle # pour désérialiser les données
import numpy as np # pour pouvoir utiliser des matrices
import matplotlib.pyplot as plt # pour l'affichage
import math
import time

start_time = time.time()

nb_neuron_input_layer       = 784 # 784 neurones en entrée
nb_neuron_per_hidden_layer  = 10
nb_neuron_output_layer      = 10 # 10 neurones en sortie pour les 10 classes [0..9]
nb_hidden_layer             = 1
nb_layers                   = nb_hidden_layer + 2
matrices                    = [] # Tableau contenant les différentes matrices
learningRate                = 0.3

# np.random.seed(1)
np.set_printoptions(suppress=True)
# np.set_printoptions(precision=20)
np.set_printoptions(precision=3)


# Initialisation des matrices associées à chaque couche (ou paire de couches)
def initialise():
    # 1ere couche
    # matrices.append(np.random.uniform(-0.5, 0.5, (nb_neuron_input_layer + 1, nb_neuron_per_hidden_layer)))
    matrices.append(np.random.uniform(-0.5, 0.5, (nb_neuron_input_layer, nb_neuron_per_hidden_layer)))
    # Itération pour chaque couche cachée
    for i in range (nb_hidden_layer - 1):
        # On créé une matrice [ taille de la couche en entrée + 1;  taille de la couche en sortie]
        # matrices.append(np.random.uniform(-0.5, 0.5, (nb_neuron_per_hidden_layer + 1, nb_neuron_per_hidden_layer)))
        matrices.append(np.random.uniform(-0.5, 0.5, (nb_neuron_per_hidden_layer, nb_neuron_per_hidden_layer)))
    # Matrice de la dernière couche cachée
    # matrices.append(np.random.uniform(-0.5, 0.5, (nb_neuron_per_hidden_layer + 1, nb_neuron_output_layer)))
    matrices.append(np.random.uniform(-0.5, 0.5, (nb_neuron_per_hidden_layer, nb_neuron_output_layer)))


def iterate(_index, _mode):

    output_vectors_array    = []
    # layer_error_array       = []
    errors                  = []

    # dans la base d'apprentissage (premier [0]), dans la base d'image (deuxième [0]), on récupère l'image à [index]
    image = data[_mode][0][_index]
    # on redimensionne l'image en 28x28
    image = image.reshape(28,28)
    # dans la base d'apprentissage ([0]), dans la base des labels ([1]), on récupère le label à [index]
    label = data[_mode][1][_index]
    # on récupère à quel chiffre cela correspond (position du 1 dans label)
    label = np.argmax(label)

    # Le vecteur d'entrée transformé en tableau 1D
    input_vector = image.flatten()

    # Feed-forward
    for j in range (len(matrices)):
        output_vector = getOutput(input_vector, matrices[j])
        output_vectors_array.append(output_vector)
        input_vector = output_vector

        # print "output_vector"
        # print output_vector

    layer_error_array = calculateError(output_vector, data[0][1][_index])
    errors.append(layer_error_array)

    # print "Error"
    # print layer_error_array

    return layer_error_array, errors, output_vectors_array



def learn(index):

    # print "matrices"
    # print matrices

    _layer_error_array, _errors, _output_vectors_array = iterate(index, 0)

    # for i in reversed(range(nb_layers)):
    for i in range(len(matrices)):

        if(i != len(matrices) and i != 0):
            # sommme                  = 0
            error_array_current     = []
            # error_array_above       = _layer_error_array
            current_output_vector   = _output_vectors_array[len(matrices) - i - 1]

            # print "matrix"
            # print matrix
            # print "###############"
            # print len(matrices)
            # print i
            # print "error_array_above"
            # print error_array_above
            
            # print "current_output_vector"
            # print current_output_vector

            # test = np.dot(matrix, error_array_above.T)
            test = current_output_vector[i] * (1 - current_output_vector[i]) * np.dot(matrices[i], np.asarray(_layer_error_array).T)
            # print "test"
            # print test
            # test2 = test[:-1]

            # _layer_error_array = test2
            _layer_error_array = test
            # _errors.append(test2)

             # 𝛿j(n) = yj(n) . [1 − yj(n)] . SOMME 𝛿(n) . wkj(n)
            # for l in range(len(current_output_vector)):
            #     sommme = 0
            #     for n in range(len(matrix[l])):
            #         sommme += matrix[l,n] * error_array_above[n]
                
            #     error_array_current.append(current_output_vector[i] * (1 - current_output_vector[i]) * sommme)

        # print "i : " + str(i)
        # print "matrices[i] : " + str(matrices[i-1].shape)
        # print "_errors : " + str(len(_errors))
        # print "_output_vectors_array[i-1] : " + str(_output_vectors_array[i-1].shape)
        matrices[i-1] += (learningRate *_layer_error_array * _output_vectors_array[i-1]).T

    # ordered_errors =  np.asarray(list(reversed(_errors)))
    # ordered_errors = list(reversed(errors))
    # ordered_outputs = np.asarray(_output_vectors_array)

    # Mise à jour des poids
    #  wji(n) = wji(n − 1) + η . 𝛿j(n) . yi(n)
    # for i in range(len(matrices)):
    #     matrices[i] += (learningRate * ordered_errors[i] *  ordered_outputs[i]).T


def test(index):

    _layer_error_array, _errors, _output_vectors_array = iterate(index, 1)

    print ">>>>>>>>>>>>>>>>>"
    print "Vecteur de sortie"
    print _output_vectors_array[len(_output_vectors_array) - 1]

    print "Vecteur attendu"
    print data[1][1][index]

    print "Erreur"
    print _layer_error_array


# Concerne la couche de sortie
def calculateError(_output, _target):
    #  𝛿i = yi . (1 − yi) . (ti − yi)
    return  _output * (1 - _output) * (_target - _output)


def getOutput(_inputVector, _matrix):

    # size                = len(_inputVector)
    # _inputVector        = np.resize(_inputVector, size + 1)
    # _inputVector[size]  = 1

    return (1 / (1 + np.exp(-np.dot(_matrix.T, _inputVector))))



# c'est ce qui sera lancé lors que l'on fait python tuto_python.py
if __name__ == '__main__':

    print "Démarrage du programme"

    # on charge les données. NB: data est une variable globale qui est donc accessible dans les fonctions au-dessus
    data = cPickle.load(gzip.open('mnist.pkl.gz'))
    print("-- Chargement des données %s seconds ---" % (time.time() - start_time))
    # on récupère le nombre d'images dans le tableau d'apprentissage
    n = np.shape(data[0][0])[0]
    print "Nb d'images " + str(n)
    # on créé un vecteur de (10,) valeurs entières prises aléatoirement entre 0 et n-1
    indices = np.random.randint(n,size=(1000,))
    # indices = np.random.randint(n,size=(1,))
    # il va valoir itérativement les valeurs dans indices / NB on aurait aussi pu écrire "for j in xrange(10): i = indices[j]"
    
    initialise()

    print "Lancement de la phase d'apprentissage"
    for i in range(100000):
    # for i in range(1):
        # print "Iteration " + str(i)
        if(i % 10000 == 0):
            print "Iteration " + str(i)
        for j in indices:
            learn(j)

    print "##########################"
    
    print "Lancement de la phase de test"
    for j in range(100):
        test(j)

print("--- %s seconds ---" % (time.time() - start_time))
