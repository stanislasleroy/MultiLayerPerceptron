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

nb_neuron_input_layer       = 784 # 784 neurones en entrée
nb_neuron_per_hidden_layer  = 10
nb_neuron_output_layer      = 10 # 10 neurones en sortie pour les 10 classes [0..9]
nb_hidden_layer             = 1
nb_layers                   = nb_hidden_layer + 2
matrixArray                 = [] # Tableau contenant les différentes matrices
learningRate                = 0.3

np.random.seed(1)
np.set_printoptions(suppress=True)
np.set_printoptions(precision=20)


# Initialisation des matrices associées à chaque couche (ou paire de couches)
def initialise():
    # 1ere couche
    matrixArray.append(np.random.uniform(-0.5, 0.5, (nb_neuron_input_layer + 1, nb_neuron_per_hidden_layer)))
    # Itération pour chaque couche cachée
    for i in range (nb_hidden_layer - 1):
        # On créé une matrice [ taille de la couche en entrée + 1;  taille de la couche en sortie]
        matrixArray.append(np.random.uniform(-0.5, 0.5, (nb_neuron_per_hidden_layer + 1, nb_neuron_per_hidden_layer)))
    # Matrice de la dernière couche cachée
    matrixArray.append(np.random.uniform(-0.5, 0.5, (nb_neuron_per_hidden_layer + 1, nb_neuron_output_layer)))


def iterate(index, mode):

    output_vectors_array    = []
    layer_error_array       = []
    errors_array            = []

    # dans la base d'apprentissage (premier [0]), dans la base d'image (deuxième [0]), on récupère l'image à [index]
    image = data[mode][0][index]
    # on redimensionne l'image en 28x28
    image = image.reshape(28,28)
    # dans la base d'apprentissage ([0]), dans la base des labels ([1]), on récupère le label à [index]
    label = data[mode][1][index]
    # on récupère à quel chiffre cela correspond (position du 1 dans label)
    label = np.argmax(label)

    # Le vecteur d'entrée transformé en tableau 1D
    input_vector = image.flatten()

    # Feed-forward
    for j in range (len(matrixArray)):
        # output_vector = getOutput(input_vector, matrixArray[j].T)
        output_vector = getOutput(input_vector, matrixArray[j])
        output_vectors_array.append(output_vector)
        input_vector = output_vector

    layer_error_array = calculateError(output_vector, data[0][1][index])
    errors_array.append(layer_error_array)

    return layer_error_array, errors_array, output_vectors_array



def learn(index):

    layer_error_array, errors_array, output_vectors_array = iterate(index, 0)

    for i in reversed(range(nb_layers)):
        if(i != len(matrixArray) and i != 0):
            sommme                  = 0
            error_array_current     = []
            matrix                  = matrixArray[i]
            error_array_above       = layer_error_array
            current_output_vector   = output_vectors_array[len(matrixArray) - i - 1]

             # 𝛿j(n) = yj(n) . [1 − yj(n)] . SOMME 𝛿(n) . wkj(n)
            for l in range(len(current_output_vector)):
                for n in range(len(matrix[l])):
                    sommme = matrix[l,n] * error_array_above[n]
                
                error_array_current.append(current_output_vector[i] * (1 - current_output_vector[i]) * sommme)
            
            layer_error_array = error_array_current
            errors_array.append(error_array_current)

    ordered_errors_array = list(reversed(errors_array)) 
    # for y, m in list(enumerate(ordered_errors_array)):
    #     print m 

    # Mise à jour des poids
    #  wji(n) = wji(n − 1) + η . 𝛿j(n) . yi(n)
    for i in range(len(matrixArray)):
        # Pour chaque matrice
        matrix = matrixArray[i].T
        # matrix = matrix.T
        error  = ordered_errors_array[i]
        output = output_vectors_array[i]

        for j in range(len(matrix)):
            for k in range(len(matrix[j])):
                new_w =  matrix[j,k] + learningRate * error[j] * output[j]
                matrix[j,k] = new_w



def test(index):

    layer_error_array, errors_array, output_vectors_array = iterate(index, 1)

    print ">>>>>>>>>>>>>>>>>"
    print "Vecteur de sortie"
    print output_vectors_array[len(output_vectors_array) - 1]

    print "Vecteur attendu"
    print data[1][1][index]

    print "Erreur"
    print layer_error_array



def calculateError(output, target):
    err = np.zeros(len(output))
    #  𝛿i = yi . (1 − yi) . (ti − yi)
    for i in range (len(output)):
        err[i] = output[i] * ( 1 - output[i]) * (target[i] - output[i])
    return err



def getOutput(inputVector, matrix):

    size                = len(inputVector)
    inputVector         =  np.resize(inputVector, size + 1)
    inputVector[size]   = 1
    m                   = matrix.T
   
    ####################
    # Avec np.dot
    # Temps pour 10 itérations de 100 images
    # Temps = 30s
    # print matrix.shape
    # print inputVector.shape
   
    resultArray1 = []
    p = np.dot(m, inputVector)
    for i in range (len(p)):
        resultArray1.append(1/(1 + np.exp(-p[i])))

    print "resultArray1"
    # return resultArray1

    ####################



    ####################
    one = np.ones(len(p))
    res1 = one / (one + np.exp(-1 * p))
    
    # print one.shape
    # res = np.exp(-1 * p)
    # print res.shape
    # print res

    # res = []
    
    print "res1"
    print res1

    # print "resultArray"
    # print resultArray

    # return resultArray
    ####################



    ####################
    # Avec np.exp
    # Temps pour 10 itérations de 100 images
    # 37 secondes sur batterie
    # Test à refaire sur secteur
    # resultArray = []
    # p = np.dot(m, inputVector)
    # res = 1 / (1 + np.exp(-1 * p) )
    res2 = 1 / (1 + np.exp(-1 * np.dot(matrix.T, inputVector)) )

    print "res2"
    print res2
    # return (1 / (1 + np.exp(-np.dot(matrix.T, inputVector))))
    ####################



    ####################
    # Code originel
    # Temps pour 10 itérations de 100 images
    # = 1m30
    # Ajout de l'entrée 1 pour chaque neurone
    # resultArray         = np.zeros(len(m))

    for i in range (len(m)):
        result = 0
        for j in range (len(m[i])):
            result += inputVector[j] * m[i,j]
        resultArray[i] = 1/(1 + math.exp(-result))

    print "resultArray"
    print resultArray

    return resultArray
    ####################
    

# c'est ce qui sera lancé lors que l'on fait python tuto_python.py
if __name__ == '__main__':

    print "Démarrage du programme"

    initialise()

    # on charge les données. NB: data est une variable globale qui est donc accessible dans les fonctions au-dessus
    data = cPickle.load(gzip.open('mnist.pkl.gz'))
    # on récupère le nombre d'images dans le tableau d'apprentissage
    n = np.shape(data[0][0])[0]
    print "Nb d'images " + str(n)
    # on créé un vecteur de (10,) valeurs entières prises aléatoirement entre 0 et n-1
    indices = np.random.randint(n,size=(1,))
    # il va valoir itérativement les valeurs dans indices / NB on aurait aussi pu écrire "for j in xrange(10): i = indices[j]"
    

    print "Lancement de la phase d'apprentissage"
    for i in range(1):
        print "Iteration " + str(i)
        for j in indices:
            learn(j)

    # print "##########################"
    
    # print "Lancement de la phase de test"
    # for j in range(100):
    #     test(j)



    # def calculateError(output, target):

    # res = []

    #  np.dot(, (target - output).T)


    # print "res"  
    # print res


    # err = np.zeros(len(output))
    # #  𝛿i = yi . (1 − yi) . (ti − yi)
    # for i in range (len(output)):
    #     err[i] = output[i] * ( 1 - output[i]) * (target[i] - output[i])
   
    # print "err"
    # print err 
    # return err
