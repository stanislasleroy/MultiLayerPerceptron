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
import cProfile

start_time = time.time()

########################
# Données statiques
nb_neuron_input_layer       = 784 # 784 neurones en entrée
nb_neuron_output_layer      = 10 # 10 neurones en sortie pour les 10 classes [0..9]

########################
# Données variables
nb_neuron_per_hidden_layer  = 10 # Valeur de départ

# nb_hidden_layer             = 1 # Valeur de départ
nb_hidden_layer             = 2

# learning_rate               = 0.3
learning_rate               = 0.1 # Valeur de départ

########################
# Variables
matrices                    = [] # Tableau contenant les différentes matrices de poids
success_rate                = 0

np.random.seed(1)
np.set_printoptions(suppress=True)
np.set_printoptions(precision=3)


# Initialisation des matrices associées à chaque couche (ou paire de couches)
def initialise():
    
    #######
    # 1ere couche

    ## Avec biais
    # matrices.append(np.random.uniform(-0.5, 0.5, (nb_neuron_input_layer + 1, nb_neuron_per_hidden_layer)))
    ## Sans biais
    matrices.append(np.random.uniform(-0.5, 0.5, (nb_neuron_input_layer, nb_neuron_per_hidden_layer)))
    
    #######
    # Itération pour chaque couche cachée
    for i in range (nb_hidden_layer - 1):
        # On créé une matrice [ taille de la couche en entrée + 1;  taille de la couche en sortie]
        ## Avec biais
        # matrices.append(np.random.uniform(-0.5, 0.5, (nb_neuron_per_hidden_layer + 1, nb_neuron_per_hidden_layer)))
        ## Sans biais
        matrices.append(np.random.uniform(-0.5, 0.5, (nb_neuron_per_hidden_layer, nb_neuron_per_hidden_layer)))
    
    #######
    # Matrice de la dernière couche cachée
    ## Avec biais
    # matrices.append(np.random.uniform(-0.5, 0.5, (nb_neuron_per_hidden_layer + 1, nb_neuron_output_layer)))
    ## Sans biais
    matrices.append(np.random.uniform(-0.5, 0.5, (nb_neuron_per_hidden_layer, nb_neuron_output_layer)))


def iterate(_index, _mode):

    output_vectors          = []
    errors                  = []
    inputs                  = []

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
    inputs.append(input_vector)

    # Feed-forward
    for j in range (len(matrices)):
        output_vector = getOutput(input_vector, matrices[j])
        output_vectors.append(output_vector)
        input_vector = output_vector.copy()
        inputs.append(input_vector)

    layer_error_array = calculateError(output_vector, data[_mode][1][_index])

    return layer_error_array, output_vectors, inputs


def test(_index):

    global success_rate

    _error_layer_sup, _output_vectors, _inputs = iterate(_index, 1)

    # print ">>>>>>>>>>>>>>>>>"
    # print "Vecteur de sortie"
    # print _output_vectors[len(_output_vectors) - 1]
    # print  "Valeur " + str(np.argmax(_output_vectors[len(_output_vectors) - 1]))
    v_1 =  np.argmax(_output_vectors[len(_output_vectors) - 1])

    # print "Vecteur attendu"
    # print data[1][1][_index]
    # print  "Valeur " + str(np.argmax(data[1][1][_index]))
    v_2 = np.argmax(data[1][1][_index])

    if(v_1 == v_2):
        success_rate += 1


def learn(index):

    _error_layer_sup, _output_vectors, _inputs = iterate(index, 0)
    
    for i in reversed(range(len(matrices))):

        if(i != 0):
            ## Avec biais
            # _output_vectors[i-1] = np.append(_output_vectors[i-1], 1)

            current_output_vector  = _output_vectors[i-1]

            # Calcul de l'erreur sur la couche courante
            # 𝛿j(n) = yj(n) . [1 − yj(n)] . ∑ 𝛿(n) . wkj(n)
            test = current_output_vector * (1 - current_output_vector) * np.dot(matrices[i], np.asarray(_error_layer_sup))
        
        ## Avec biais
        # _inputs[i] = np.append(_inputs[i], 1)

        # Mise à jour des poids
        # --> Utilisation du produit de Kronecker
        # wji(n) = wji(n − 1) + η . 𝛿j(n) . yi(n)
        matrices[i] += learning_rate * np.kron(_inputs[i].T, _error_layer_sup).reshape((len(_inputs[i].T), len(_error_layer_sup)))
        
        # Avec biais
        # _error_layer_sup = test[:-1]

        # Sans biais
        _error_layer_sup = test


# Concerne la couche de sortie
#  𝛿i = yi . (1 − yi) . (ti − yi)
def calculateError(_output, _target):
    return  _output * (1 - _output) * (_target - _output)


def getOutput(_input, _matrix):
    ## Avec biais
    # _input = np.append(_input, 1)

    return (1 / (1 + np.exp(-np.dot(_input.T, _matrix))))


if __name__ == '__main__':

    print "Démarrage du programme"

    # on charge les données. NB: data est une variable globale qui est donc accessible dans les fonctions au-dessus
    data = cPickle.load(gzip.open('mnist.pkl.gz'))
    time_1 = time.time()
    print("-- Chargement des données %s secondes ---" % (time.time() - start_time))
    # on récupère le nombre d'images dans le tableau d'apprentissage
    n = np.shape(data[0][0])[0]
    print "Nb d'images " + str(n)
    # on créé un vecteur de (10,) valeurs entières prises aléatoirement entre 0 et n-1
    indices = np.random.randint(n,size=(62900,))
    # il va valoir itérativement les valeurs dans indices / NB on aurait aussi pu écrire "for j in xrange(10): i = indices[j]"
    
    initialise()
    print "Nb de couches cachées de la phase d'apprentissage : " + str(nb_hidden_layer)
    print "Nb de neurones par couche cachée : " + str(nb_neuron_per_hidden_layer) 
    print "Pas d'apprentissage : " + str(learning_rate) 

    # Apprentissage
    print "Lancement de la phase d'apprentissage"

    for j in xrange(62000):
        i = indices[j]
        if(j % 10000 == 0):
            print "Iteration " + str(j)
        learn(i)
        #  cProfile.run('learn(' + str(i) + ')')

    print("--- Apprentissage %s secondes ---" % (time.time() - time_1))
    
    # Test
    print "##########################"

    nb_of_item = 1000

    n_test = np.shape(data[1][0])[0]
    indices_test = np.random.randint(n_test,size=(nb_of_item,))
    
    print "Lancement de la phase de test"
    for j in indices_test:
    # for j in xrange(1000):
        # print "Image " + str (j)
        test(j)

    value = (success_rate/float(nb_of_item)) * 100
    print "Taux de réussite " + str(value)

    # Durée totale 
    print("--- Total %s secondes ---" % (time.time() - start_time))
