# A propos de moi
Salut, je m'appelle Mountassir.

Je poursuie actuellement un double diplome d'ingénieur(Informatique et télécommunications - Mathématiques appliqués) sur la thématique de l'intelligence artificielle hybride.

Dans le cadre de ma formation (formation en alterance), j'ai eu l'opportunité de travailler chez l'entreprise Akanthas en tant que "Data scientist and developer".

# Ma vision et convictions
En essayant d'apporter le maximum de valeur à Akanthas, j'ai amélioré et acquis beaucoup de compétances personnelles et techniques, mais j'ai aussi développé une vision et une conviction.

J'ai la conviction qu'un bon data scientist est avant tout, une personne curieuse, qui pose beaucoup de questions afin de bien comprendre le besoin et qui trouve ensuite les meilleurs solutions à implémenter pour y répondre en minimisant les coûts (temps et argent).

Pour trouver les meilleurs solutions, le data scientist doit être à jour avec les nouveautés de son domaine (Les dernières technologies, modèles et documents de recherche). Imaginons que vous avez demandé à deux data scientist de créer un modèle de classification et de l'entrainer pour distinguer entre differentes classes d'images.
* Le premier data scientist programme un modèle en choisisant l'architecture par lui même, donc il aura besoin de plus de temps (pour optimiser l'architecture avec du fine-tuning) et n'aura à la fin qu'une précision de 80% (chiffre donné à titre d'exemple).
* Le deuxieme data scientist est au courant que le meilleur modèle de classification d'image (à l'heure que j'écris ce document) est le **Vision Transformer ViT** (Voici un <u><a href="https://arxiv.org/pdf/2010.11929v2.pdf" target="_blank">document de recherche</a></u> que je trouve intéressant) et donc en implémentant ce modèle state of the art, il a réussi à répondre au besoin dans moins de temps mais avec une meilleur précision.

Lequel selon vous est le meilleur data scientist ?

# Projets
Dans cette section vous trouverez une description des projets sur lequels j'ai travaillé.



## <u>I - Vision Par Ordinateur</u>
###  <u>1. Classification d'image</u>
 
La **Classification d'image** est la branche de la **vision par ordinateur** qui consiste à classifier une image selon son contenu.  
**Example :**  Détérminer si une image contient un chien ou un chat (sans préciser sa position dans l'image).  

Dans le dossier **Computer Vision** ➡ **Image Classification** vous trouverez les projets suivants : 

* **mnist**: l'objectif de ce projet est de faire la reconnaissance de chiffres manuscrits, nous avons implémenté et comparé différentes approches, notamment l'algorithme des k plus proches voisins, les arbres de décision, les machine à support de vecteur et les réseaux de neurones. Pour les réseaux de neurones nous avons utilisé une architecture réduite de Xception (Xception est une très bonne architecture pour les tâches de classification). Nous avons réussi à atteindre une précision de 98.98% (almost 99%).

* **covid**: l'objectif de ce projet est de décider à partir de l'image x-ray des poumons d'une personne, si cette personne a le covid, la pneumonie ou s'elle est saine. dans un premier temps, nous avons implémenté un modèle de réseaux de neurones simples (précision faible : 20%), puis nous avons ajouté à ce dernier une couche VGG16 d'**extraction de caractéristiques (feature extraction)**, avec cette dernière notre modèle a atteint une précision de plus de 88% en moins d'epochs, ce qui montre l'intérêt de l'extraction des caractéristiques. Finalement nous avons implementé le fameux **EfficientNetB2** qui nous a permis d'atteindre (même sans feature extraction) une précision de 91%.

###  <u>2. Détéction d'objet ( Object / Instance Detection )</u>

La **Détéction d'Objet** est la  branche de la **vision par ordinateur** qui consiste à détécter les objets dans une image.
**Example :**  Détécter les oiseaux dans une image et entourer chaqu'un d'entre eux par un rectangle (On connait exactement la position de l'objet dans l'image).  

Dans le dossier **Computer Vision** ➡ **Object Detection** vous trouverez les projets suivants :

* **mask** : l'objectif de ce projet est de détécter dans une image (ou video) les personnes avec masque, sans masque et ceux qui portent un masque de manière incorrect. Nous avons entrainé le modèle <b>Faster RCNN X101 FPN 3x</b> qui est l'un des plus puissants modèles de détection d'objet implémentés par détection2. Nous avons réussi à atteindre une précision de 

* **signLanguage** : 

