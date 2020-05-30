# Covid-19 Chest X-ray Detector
## Overview

**Avertissement** : Ceci est un projet d'étude et ne peut en aucun cas être utiliser dans la vraie vie.

Modèle de Deep Learning construit en utilisant un Réseau Neuronal Convulsif (CNN), afin de détecter par l'intermédiare de radiographie du thorax les pathologie de COVID-19. Les données (radiographoes du thorax) sont issues d'images open-source de patient atteints de COVID-19, d'infection pulmonaires autres et de patients sains.

Ce projet vise à détecter automatiquement les patients atteint de COVID-19 dans l'ensemble des données que nous avons recueiili. 

Nous avons choisi d'utiliser un  [Réseau Neuronal Convultif (CNN)](https://fr.wikipedia.org/wiki/R%C3%A9seau_neuronal_convolutif), pour le traitement des images sur lequel nous avons utilisé des poids pré-formés avec un modèle [VGG19](https://gist.github.com/baraldilorenzo/8d096f48a1be4a2d660d). 

## Données

Les données pour ce projet sont issues de [covid-chestxray-dataset](https://github.com/ieee8023/covid-chestxray-dataset) pour la partie COVID-19 et autres infections pulmonaires et du dataset de [chest-xray-pneumonia](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia) pour la partie radios "normales".

### Projet COVID-19-chestxray-data

Du projet COVID-19-chestxray-data nous avons extraits les radios qui sont prises en postéro-antérieures. Cela représente 195 observations

![](img/COVID-19-chestxray-data.png)


Les radios concernent différentes affections qui sont étiquetées comme suit :  

```
COVID-19          141
COVID-19, ARDS     11
Streptococcus      13
Pneumocystis       12
SARS                8
ARDS                4
Legionella          2
No Finding          2
Klebsiella          1
Chlamydophila       1
```

![](img/finding_distribution.png)

Nous avons regroupé les affections étiquetées COVID19 et COVID-19, ARDS en une seule étiquette covid-19 et les autres affections ont été classées en "Normal". Ce qui fait 152 observations où le COVID-19 est détecté et 43 qui sont considérées comme normales.

Pour que les données soient balancées, nous avons ajouté des données en provenance d'un autre dataset

### Projet chest-xray-pneumonia

Du projet chest-xray-pneumonia, nous avons extrait 109 radios de poitrine qui sont etiquetées comme "normal".

![](img/chest-xray-pneumonia.png)

### Organisation des données

Ainsi nous avons 152 radios étiquetées COVID-19 et 152 étiquetées "Normale", sur les 152 normales il y 111 radio sans pathologie detectée  et 41 avec une pathologie pulmonaire.

Pour plus de facilité, nous avons copié les données de chaque dataset dans de nouveaux répertoires afin de faciliter l'entrainement du modèle.

* un répertoire **train** avec 2 sous-répertoire **0_normal** et **1_covid**
* un répertoire **test** avec 2 sous-répertoire **0_normal** et **1_covid** 
* un répertoire **predict** avec des radios pulmonires à prédire. Pour plus de facilité, nous avons prefixé le nom des fichiers de ce répertoire par la valeur à prédire (0 ou 1) 

## Traitement des images
Avant que les images ne soient entrainées par le modèle, nous avons normalisé les images en transformant les valeurs RGD qui sont comprises entre 0 et 255 par un changement d'échelle de 1/255 pour obtenit des valeurs entre 0 et 1.
Les images sont redimentionnées en 320x382 pixels.

## Modèle

Nous avons choisi d'utiliser un  [Réseau Neuronal Convultif (CNN)](https://fr.wikipedia.org/wiki/R%C3%A9seau_neuronal_convolutif), pour le traitement des images sur lequel nous avons utilisé des poids pré-formés avec un modèle [VGG19](https://gist.github.com/baraldilorenzo/8d096f48a1be4a2d660d). 

A ce modèle nous avons ajouté les couches suivantes :
 - AveragePooling2D
 - Ajout d'un nouveau canal 
 - Couche d'activation 
 - Suppression arbitraire
 - Couche d'activation finale

```
Model: "model_4"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_5 (InputLayer)         (None, 320, 382, 3)       0         
_________________________________________________________________
block1_conv1 (Conv2D)        (None, 320, 382, 64)      1792      
_________________________________________________________________
block1_conv2 (Conv2D)        (None, 320, 382, 64)      36928     
_________________________________________________________________
block1_pool (MaxPooling2D)   (None, 160, 191, 64)      0         
_________________________________________________________________
block2_conv1 (Conv2D)        (None, 160, 191, 128)     73856     
_________________________________________________________________
block2_conv2 (Conv2D)        (None, 160, 191, 128)     147584    
_________________________________________________________________
block2_pool (MaxPooling2D)   (None, 80, 95, 128)       0         
_________________________________________________________________
block3_conv1 (Conv2D)        (None, 80, 95, 256)       295168    
_________________________________________________________________
block3_conv2 (Conv2D)        (None, 80, 95, 256)       590080    
_________________________________________________________________
block3_conv3 (Conv2D)        (None, 80, 95, 256)       590080    
_________________________________________________________________
block3_conv4 (Conv2D)        (None, 80, 95, 256)       590080    
_________________________________________________________________
block3_pool (MaxPooling2D)   (None, 40, 47, 256)       0         
_________________________________________________________________
block4_conv1 (Conv2D)        (None, 40, 47, 512)       1180160   
_________________________________________________________________
block4_conv2 (Conv2D)        (None, 40, 47, 512)       2359808   
_________________________________________________________________
block4_conv3 (Conv2D)        (None, 40, 47, 512)       2359808   
_________________________________________________________________
block4_conv4 (Conv2D)        (None, 40, 47, 512)       2359808   
_________________________________________________________________
block4_pool (MaxPooling2D)   (None, 20, 23, 512)       0         
_________________________________________________________________
block5_conv1 (Conv2D)        (None, 20, 23, 512)       2359808   
_________________________________________________________________
block5_conv2 (Conv2D)        (None, 20, 23, 512)       2359808   
_________________________________________________________________
block5_conv3 (Conv2D)        (None, 20, 23, 512)       2359808   
_________________________________________________________________
block5_conv4 (Conv2D)        (None, 20, 23, 512)       2359808   
_________________________________________________________________
block5_pool (MaxPooling2D)   (None, 10, 11, 512)       0         
_________________________________________________________________
average_pooling2d_4 (Average (None, 2, 2, 512)         0         
_________________________________________________________________
flatten_6 (Flatten)          (None, 2048)              0         
_________________________________________________________________
dense_11 (Dense)             (None, 128)               262272    
_________________________________________________________________
dropout_4 (Dropout)          (None, 128)               0         
_________________________________________________________________
dense_12 (Dense)             (None, 1)                 129       
=================================================================
Total params: 20,286,785
Trainable params: 4,982,017
Non-trainable params: 15,304,768
_________________________________________________________________
```



![Loss and accuracy](./img/model-loss-accuracy.png)
## Prédiction
Avec les images présentent dans le répertoire "predict", nous avons confronté notre modèle.
Ce réptoire contient 31 images, dont 16 images identifiées comme atteint de COVID-19. Sur les 15 images considérées comme normale 3 sont issues du 2ème jeu de données.

La matrice de confusion donne les éléments suivants :

![Confusion matrix](./img/confusion_matrix.png)

sur 15 cas négatif, 1 n'est pas correctement identifié et sur 16 cas positif, 1 cas n'ai  pas correctement identifié.

```
[[14  1]
 [ 1 15]]
              precision    recall  f1-score   support

           0       0.93      0.93      0.93        15
           1       0.94      0.94      0.94        16

    accuracy                           0.94        31
   macro avg       0.94      0.94      0.94        31
weighted avg       0.94      0.94      0.94        31
```

## Conclusion
Ce modèle présente une performance honorable, il faudrait cependant ajouter plus de données pour vérifier sa réelle efficacité. Une piste d'amélioration serait d'ajouter des données clinique (taux Leucocyte, température, etc.) à  l'analyse des images (comme dans cet [exemple](https://cloud.google.com/blog/products/ai-machine-learning/how-20th-century-fox-uses-ml-to-predict-a-movie-audience)).

## References

```
Joseph Paul Cohen and Paul Morrison and Lan Dao
COVID-19 image data collection, arXiv:2003.11597, 2020
https://github.com/ieee8023/covid-chestxray-dataset
```

