# -*- coding: utf-8 -*-
# %% [markdown]
# # Évaluation de python-numérique
#
# ***Analyse morphologique de défauts 3D***
#
# ***

# %% [markdown]
# ## Votre identité

# %% [markdown]
# Ne touchez rien, remplacez simplement les ???
#
# Prénom: Jeanne
#
# Nom: Bessoud
#
# Langage-avancé (Python ou C++): Python
#
# Adresse mail: jeanne.bessoud@mines-paris.fr
#
# ***

# %% [markdown]
# ## Quelques éléments de contexte et objectifs
#
# Vous allez travaillez dans ce projet sur des données réelles concernant des défauts observés dans des soudures. Ces défauts sont problématiques car ils peuvent occasionner la rupture prématurée d'une pièce, ce qui peut avoir de lourdes conséquences. De nombreux travaux actuels visent donc à caractériser la **nocivité** des défauts. La morphologie de ces défauts est un paramètre qui influe au premier ordre sur cette nocivité.
#
# Dans ce projet, vous aurez l'occasion de manipuler des données qui caractérisent la morphologie de défauts réels observés dans une soudure volontairement ratée ! Votre mission est de mener une analyse permettant de classer les défauts selon leur forme : en effet, deux défauts avec des morphologies similaires auront une nocivité comparable. 

# %% [markdown]
# ### Import des librairies numériques
#
# Importez les librairies numériques utiles au projet, pour le début du projet, il s'agit de `pandas`, `numpy` et `pyplot` de `matplotlib`.

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# %% [markdown]
# ## Lecture des données
#
# Les données se trouvent dans le fichier `defect_data.csv`. Ce fichier contient treize colonnes de données :
# * la première colonne est un identifiant de défaut (attention, il y a 4040 défauts mais les ids varient entre 1 et 4183) ;
# * les neuf colonnes suivantes sont des descripteurs de forme sur lesquels l'étude va se concentrer ;
# * les trois dernières colonnes sont des indicateurs mécaniques auxquels nous n'allons pas nous intéresser dans ce projet.
#
# Lisez les données dans une data-frame en ne gardant que les 9 descripteurs de forme et en indexant vos lignes par les identifiants des défauts. Ces deux opérations doivent être faites au moment de la lecture des données.
#
# Affichez la forme et les deux dernières lignes de la data-frame.

# %%
df = pd.read_csv("defect_data.csv", index_col='id', usecols=range(0,10))
print(df.shape)
df.tail(2)
df


# %% [markdown]
# ### Parenthèse sur descripteurs morphologiques 
#
# **Note: cette section vous donne du contexte quant à la signification des données que vous manipulez et la façon dont elles ont été acquises. Si certains aspects vous semblent nébuleux, cela ne vous empêchera pas de finir le projet !**
#
# Vous allez manipuler dans ce projet des descripteurs morphologiques. Ces descripteurs sont ici utilisés pour caractériser des défauts, observés par tomographie aux rayons X dans des soudures liant deux pièces métalliques. La tomographie consiste à prendre un jeu de radiographies (comme chez le médecin, avec un rayonnement plus puissant) en faisant tourner la pièce entre chaque prise de vue. En appliquant un algorithme de reconstruction idoine à l'ensemble des clichés, il est possible de remonter à une image 3D des pièces scannées. Plus la zone que l'on traverse est dense plus elle est claire (comme chez le médecin : vos os apparaissent plus clair que vos muscles). Dans notre cas, le constraste entre les défauts constitués d'air et le métal est très marqué : on observe donc les défauts en noir et le métal en gris. Un défaut est donc un amas de voxels (l'équivalent des pixels pour une image 3D) noirs. Sur l'image ci-dessous, les défauts ont été extraits et sont représentés en 3D par les volumes rouges. 
#
# <img src="media/defects_3D.png" width="400px">
#
# Vous voyez qu'ils sont nombreux, de taille et de forme variées. À chaque volume rouge que vous observez correspond une ligne de votre `DataFrame` qui contient les descripteurs morphologiques du-dit défaut. 
#
#
# #### Descripteur $r_1$ (`radius1`)
# En notant $N$ le nombre de voxels constituant le défaut, on obtient le volume du défaut $V=N\times v_0$ (où $v_0$ est le volume d'un voxel).On peut alors définir son *rayon équivalent* comme le rayon de la sphère de même volume soit :
# \begin{equation*}
#  R_{eq} = \left(\frac{3V}{4\pi}\right)^{1/3}
# \end{equation*}
#
# On définit ensuite le *rayon moyen* $R_m$ du défaut comme la moyenne sur tous les voxels de la distance au centre de gravité du défaut. 
#
# $R_{eq}$ et $R_m$ portent une information sur la taille du défaut. En les combinant comme suit:
# \begin{equation*}
#  r_1 = \frac{R_{eq} - R_m}{R_m}
# \end{equation*}
# on la fait disparaître : $r_1$ vaut 1/3 pour une sphère quel que soit son rayon. 
#
# **Note :** vous aurez remarqué que $r_1$ est donc sans dimension.
#
# #### Descripteurs basés sur la matrice d'inertie ($\lambda_1$ et $\lambda_2$) (`lambda1`, `lambda2`)
# La matrice d'inertie de chaque défaut est calculée. Pour ce faire, on remplace tout simplement les intégrales sur le volume présentes dans les formules que vous connaissez par une somme sur les voxels. Par exemple:
# \begin{equation}
#  I_{xy} = -\sum\limits_{v\in\rm{defect}} (x(v)-\bar{x})(y(v)-\bar{y})\qquad \text{avec } \bar{x} = \frac{1}{N}\sum\limits_{v\in\rm{defect}} x(v) \text{ et } \bar{y} = \frac{1}{N}\sum\limits_{v\in\rm{defect}} y(v)
# \end{equation}
# Cette matrice est symétrique réelle, elle peut donc être diagonalisée. Les trois valeurs propres obtenues $I_1 \geq I_2 \geq I_3$ sont les moments d'inertie du défaut dans son repère principal d'inertie. Ces derniers portent de manière intrinsèque une information sur le volume du défaut. Pour la faire disparaître, il suffit de normaliser ces moments comme suit : 
#
# \begin{equation}
#  \lambda_i = \frac{I_i}{I_1+I_2+I_3}
# \end{equation}
#
# On obtient alors trois indicateurs $\lambda_1 \geq \lambda_2 \geq \lambda_3$ vérifiant notamment $\lambda_1 + \lambda_2 + \lambda_3 = 1$ ce qui explique que l'on ne garde que les deux premiers. En utilisant les propriétés des moments d'inertie, on peut montrer que les points obtenus se situent dans le triangle formé par $(1/3, 1/3)$, $(1/2, 1/4)$ et $(1/2, 1/2)$ dans le plan $(\lambda_1, \lambda_2)$. Vous pourrez vérifier cela dans la suite ! 
#
# La position du point dans le triangle renseigne sur sa forme *globale*, comme indiqué par l'image suivante : 
# ∑
# <img src="media/l1_l2.png" width="400px">
#
# #### Convexité (`convexity`)
#
# L'indicateur de convexité utilisé est simplement le rapport entre le volume du défaut et de son convexe englobant. $C = V/V_{CH} \leq 1$. Lorsque qu'un défaut est convexe, il est égal à son convexe englobant et donc $C$ vaut 1.
#
# #### Sphéricité (`sphericity`)
#
# L'indicateur de sphéricité permet de calculer l'écart d'un défaut à une sphère. On utilise ici la caractéristique de la sphère qui minimise la surface extérieure pour un volume donné. La grandeur : 
# \begin{equation*}
# I_S = \frac{6\sqrt{\pi}V}{S^{3/2}}
# \end{equation*}
# où $V$ est le volume du défaut et $S$ sa surface vaut 1 pour une sphère et est inférieur à 1 sinon. 
#
# #### Indicateurs basés sur la mesure de la courbure (`varCurv`, `intCurv`)
#
# Les deux courbures principales $\kappa_1$ et $\kappa_2$ sont calculées en chaque point de la surface des défauts ([ici pour les plus curieux](https://fr.wikipedia.org/wiki/Courbure_principale)). Ces courbures permettent de caractériser la forme locale du défaut. Elle sont combinées pour calculer la courbure moyenne $H = (\kappa_1+\kappa_2)/2$ et la courbure de Gauss $\gamma = \kappa_1\kappa_2$. Pour s'affranchir de l'information sur la taille (pour une sphère de rayon $R$, on a en tout point $\kappa_1 = \kappa_2 = 1/R$), les défauts sont normalisés en volume avant d'en calculer les courbures.
# Les indicateurs retenus sont les suivants:
#
#  - la variance de la courbure de Gauss (colonne `varCurv`) $Var(H)$ ; 
#  - l'intégrale de $\gamma$ sur la surface du défaut(colonne `intCurv`) $\int_S \gamma dS$. 
#  
# Ces indicateurs valent respectivement $0$ et $4\pi$ pour une sphère.
#
# #### Indicateurs sur la boite englobante $(\beta_1, \beta_2)$ (`b1`, `b2`) 
#
# Finalement, c'est une information sur la boite englobante du défaut dans son repère principal d'inertie qui est cachée dans $(\beta_1, \beta_2)$. En notant $B_1, B_2, B_3$ les dimensions (décroissantes) de la boite englobante, on réalise la même normalisation que pour les moments d'inertie : 
# \begin{equation}
#  \beta_i = \frac{B_i}{B_1+B_2+B_3}
# \end{equation}
#
# ***

# %% [markdown]
# ## Visualisation des défauts
#
# Pour que vous saissiez un peu mieux la signification des descripteurs morphologiques, nous avons concocté une petite fonction utilitaire qui vous permettra de visualiser les défauts. Pour que vous puissiez interagir avec `pyplot`, il nous est imposé de changer le backend avec la commande `%matplotlib notebook` et de recharger le module. Pour revenir dans le mode de visualisation précédent, vous devrez évaluer la cellule qui contient la commande `%matplotlib inline` qui arrive un peu plus tard !  
#
# *Nous n'avons malheureusement pas trouvé de solution pour que ce changement soit transparent pour vous... :(*
#
# Amusez vous à chercher des défauts **extrêmes** pour comprendre. Par exemple le défaut qui maximise $\lambda_2$ sera celui qui a la forme la plus *filaire* alors que celui qui minimise aura la forme la plus *plate*. Pourquoi ne pas jeter un coup d'oeil au défaut le moins convexe ? 

# %%
# Ne touchez pas à ça c'est pour la visualisation interactive ! 
# %matplotlib notebook
from importlib import reload
from utilities import plot_defect
reload(plt)
# À partir de maintenant vous pouvez vous amuser ! 

# On récupère un id intéressant
id_to_plot = df.index[df['lambda1'].argmax()]
# On affiche à l'écran les valeurs de ses descripteurs
print(df.loc[id_to_plot])
# On l'affiche
plot_defect(id_to_plot)

# %% [markdown]
# N'oubliez pas d'aller rendre visite au défaut avec l'id `4022` qui a une forme rigolote, avec ses petites excroissances. 

# %%
# Le défaut 4022 ! 
print(df.loc[4022])
plot_defect(4022)

# %% [markdown]
# On vous parlait juste avant de défauts de morphologie proche ! Et si une simple distance euclidienne en dimension 9 fonctionnait ? Calculez le défaut le plus proche du défaut `4022` dans l'espace de dimension 9, et tracez-le ! Se ressemblent-ils ?

# %%
ligne_4022 = df.loc[4022]
s_min = 0
for j in range (9): #on definit la distance euclidienne min initiale comme celle entre le défaut 1 et le 4022 
    s_min += (ligne_4022.iloc[j] - df.iloc[0, j]) ** 2
d_min = np.sqrt(s_min)
i_min = 0

for k in range (1, 4040): #on parcourt tout les défauts 
    s = 0 
    if k != 4022 : #on exclu le defaut 4022 ou la distance à lui-meme est nulle 
        for j in range (9):
            s += (ligne_4022.iloc[j] - df.iloc[k, j]) ** 2
            distance = np.sqrt(s)
            if distance < d_min :
                d_min = distance
                i_min = k
print(i_min)


# %% [markdown]
# **Eh non!** Le défaut le plus proche du défaut `4022` est une patatoïde quelconque. Deux explications sont possibles :
#  * soit la distance euclidienne n'est pas pertinente ici ;
#  * soit le défaut `4022` est le seul avec des petites excroissances... 
# Je vous laisse aller voir le défaut `796` pour trancher entre les deux propositions.. 
#

# %%
print(df.loc[796])
plot_defect(796)

# %%
# On revient en mode de visu standard après avoir évalué cette cellule
# %matplotlib inline
reload(plt)


# %% [markdown]
# ## Visualisation des données
#
# Avant de commencer toute analyse à proprement parler de données, il est nécessaire de passer un moment à les observer.
#
# Pour ce faire, nous allons écrire des fonctions utilitaires qui se chargeront de tracer des graphes classiques.

# %% [markdown]
# ### Tracé d'un histogramme
#
# Écrire une fonction `histogram` qui trace l'histogramme d'une série de points. 
#
#
# Par exemple l'appel `histogram(df['radius1'], nbins=10)` devrait renvoyer quelque chose qui ressemble à ceci:
#
# <img src="media/defects-histogram.png" width="400px">

# %%
def histogram(serie, nbins) :
    """
    returns the histogram of a series of point, regrouping the values around nbins values 
    """
    serie.plot.hist(nbins)
    plt.show()
 

    return(serie.plot.hist(bins=nbins))


# %%
# pour vérifier
histogram(df['radius1'], nbins=10)

# %%
# pour vérifier
# c'est bien si votre fonction marche aussi avec une dataframe
histogram(df[['radius1']], nbins=10)


# %% [markdown]
# #### *Bonus* : un histogramme adaptable aux goûts de chacun
#
# Modifier la fonction `histogram` pour que l'utilisateur puisse préciser par exemple: le nom des étiquettes des axes, les couleurs à utiliser pour représenter l'histogramme...
#
# Par exemple en appelant
# ```python
# histogram2(df['radius1'], nbins=10,
#        xlabel='radius1', ylabel='occurrences',
#        histkwargs={'color': 'red'})
# ```
#
# on obtiendrait quelquechose comme ceci
#
# <img src="media/defects-histogram2.png" width="400px">

# %%
def histogram2(serie, nbins, xlabel, ylabel, hist_kwargs) :
    """
    returns the histogram of a series of point, regrouping the values around nbins values 
    """
    L=[]
    for k in range(4040):
        L.append(serie.iloc[k])
    plt.hist(L, nbins, color=hist_kwargs['color'])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()



# %%
# pour vérifier

# seulement si la fonction est définie
if 'histogram2' in globals():
    histogram2(df['radius1'], nbins=10,
           xlabel='radius1', ylabel='occurrences',
           hist_kwargs={'color': 'red'})
else:
    print("vous avez choisi de ne pas faire le bonus")


# %% [markdown]
# ### Tracé de nuages de points
#
# Écrire une fonction `correlation_plot` qui permet de tracer le nuage de points entre deux séries de données. 
# L'appel de cette fontion comme suit `correlation_plot(df['lambda1'], df['lambda2'])` devrait donner une image ressemblant à celle-ci :
#
# Ces tracés illustrent le *degré de similarité* des colonnes. Notons, qu'il existe des indices de similarité comme par exemple: la covariance, le coefficient de corrélation de Pearson...
#
#
# <img src="media/defects-correlation.png" width="400px">

# %%
def correlation_plot(s1,s2):
    """
    returns the correlation between two series of point s1 and s2, drawing s1 on the x-axis and s2 on the y-axis
    """
    plt.scatter(s1,s2, color='black', s=5)
    plt.show()


# %%
# pour vérifier
correlation_plot(df['lambda1'], df['lambda2'])


# %% [markdown]
# #### *Bonus* les nuages de points pour l'utilisateur casse-pieds (ou daltonien ;) )
# Modifier la fonction `correlation_plot` pour que l'utilisateur puisse préciser des noms pour les axes, et choisir l'aspect des points tracés (couleur, taille, forme, ...). 
#
# par exemple en appelant
# ```python
# correlation_plot2(df['lambda1'], df['lambda2'], 
#                   xlabel='lambda1', ylabel='lambda2',
#                   plot_kwargs={'marker': 'x', 'color': 'red'})
# ```
# on obtiendrait quelque chose comme
#
# <img src="media/defects-correlation2.png" width="400px">

# %%
def correlation_plot2(s1, s2, xlabel, ylabel, plot_kwargs):
    """
    returns the correlation between two series of point s1 and s2, drawing s1 on the x-axis and s2 on the y-axis,
    and representing every point with a red cross
    """
    plt.scatter(s1, s2, marker = plot_kwargs['marker'], color = plot_kwargs['color'])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()



# %%
# pour vérifier 

# seulement si la fonction est définie
if 'correlation_plot2' in globals():
    correlation_plot2(df['lambda1'], df['lambda2'], 
                      xlabel='lambda1', ylabel='lambda2',
                      plot_kwargs={'marker': 'x', 'color': 'red'})
else:
    print("vous avez choisi de ne pas faire le bonus")    


# %% [markdown]
# #### *Bonus 2* Tracer le triangle d'inertie en plus
#
# On vous disait plus tôt que les points dans le plan $(\lambda_1, \lambda2)$ sont forcément dans le triangle formé par $(1/3, 1/3)$, $(1/2, 1/4)$ et $(1/2, 1/2)$. 
# Essayez de superposer les données au triangle pour mettre cela en évidence ! 
# Le résultat pourrait ressembler à ceci :
#
# <img src="media/defects-correlation3.png" width="400px">
#
# (Vous pouvez faire ce bonus sans avoir fait le précédent)

# %%
def triangle_inertie (s1, s2, xlabel, ylabel):
    """
    returns the correlation between two series of point s1 and s2 with the triangle of inertia 
    """
    X = [1/3, 1/2, 1/2, 1/3]
    Y = [1/3, 1/4, 1/2, 1/3]
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.plot(X, Y, linestyle = 'dashed', color = 'red')
    correlation_plot(s1, s2)
    plt.show()

triangle_inertie(df['lambda1'], df['lambda2'],'lambda1', 'lambda2')


# %% [markdown]
# ### Affichage de tous les plots des colonnes
#
# Écrire une fonction `plot2D` qui prend en argument une dataframe et qui affiche 
# * les histogrammes des colonnes
# * les plots des corrélations des couples des colonnes  
# n'affichez qu'une seule fois les corrélations par couple de colonnes
#
# avec `plot2D(df[['radius1', 'lambda1', 'lambda2']])` vous devriez obtenir quelque chose comme ceci
#
# <img src="media/defects-plot2d.png" width="200px">

# %%
def plot2D(df):
    """
    returns all the histograms of the columns of a dataframe and the correlation between two columns, without 
    repetition of the correlation of the couple of columns
    """
    n = len(list(df.columns)) #nombre de colonnes 
    for colonne in df.columns :
        print(histogram2(df[colonne], nbins = 30, xlabel = colonne, ylabel = 'frequency', hist_kwargs= {'color': 'blue'}))
    n = len(list(df.columns)) #nombre de colonnes 
    for i in range(n-1):
        for j in range(i+1, n):
            print(correlation_plot2(df[df.columns[i]], df[df.columns[j]], xlabel = df.columns[i], ylabel = df.columns[j], plot_kwargs = {'marker': 'x', 'color': 'black'}))



# %%
# pour corriger

plot2D(df[['radius1', 'lambda1', 'lambda2']])


# %% [markdown]
# #### *Bonus++* (dataviz expert-level) le tableau des plots des colonnes
#
# Écrire une fonction `scatter_matrix` qui prend une dataframe en argument et affiche un tableau de graphes avec
# * sur la diagonale les histogrammes des colonnes
# * dans les autres positions les plots des corrélations des couples des colonnes
#
# avec `scatter_matrix(df[['radius1', 'lambda1', 'b2']], nbins=100, hist_kwargs={'fc': 'g'})` vous devriez obtenir à peu près ceci
#
# <img src="media/defects-matrix.png" width="500px">

# %%
def scatter_matrix(df, nbins, hist_kwargs):
    """
    returns a tab of graphs of a dataframe df, where all the histograms of the columns are on the diagonal. 
    On each line are drawn the correlations between the column used for the histogram and the another columns, 
    without repetition of the correlation of the couple of columns.
    """
    n = len(list(df.columns)) #nombre de colonnes 
    fig, axs = plt.subplots(n, n, figsize=(15,15))
    for i in range(n) : #on rempli la diagonale du tableau avec les histogrammes
        axs[i, i].hist(df.iloc[:,i], nbins, color=hist_kwargs['color'])
        for j in range (n): #on parcourt les autres colonnes
            if j != i : 
                axs[i, j].scatter(df.iloc[:,i], df.iloc[:,j], s=1)
                
    

# %%
# pour vérifier 

scatter_matrix(df[['radius1', 'lambda1', 'b2']], nbins=100, hist_kwargs={'color': 'g'})

# %% [markdown]
# ### Corrélations entre les données
#
# Utilisez les fonctions que vous venez d'implémenter pour rechercher (visuellement) les meilleures correlations qui ressortent entre les différentes caractéristiques morphologiques.
#
# Plottez la corrélation qui vous semble la plus frappante (i.e. la plus informative), motivez votre choix.
#

# %% [markdown]
# Réponse : 
# On observe toutes les correlations avec scatter_matrix(df[df.columns], nbins=100, hist_kwargs={'color': 'g'})

# %%
# on observe toutes les correlations : scatter_matrix(df[df.columns], nbins=100, hist_kwargs={'color': 'g'})

#la meilleur correlation semble etre celle entre la 4eme et la 5eme colonne, pusiqu'on voit apparaitre une droite linéaire sur le graphe de corrélation

scatter_matrix(df[['convexity','sphericity']], nbins=100, hist_kwargs={'color': 'g'}) 


# %% [markdown]
# ## Analyse en composantes principales (ACP)
#
# Les corrélations entre variables mises en évidence précédemment nous indiquent que certaines informations, apportées par les indicateurs, sont parfois redondantes.
#
# L'analyse en composantes principales est une méthode qui va permettre de construire un jeu de *composantes principales* qui sont des combinaisons linéaires des caratéristiques. Ces composantes principales sont indépendantes les unes des autres. Ce type d'analyse est généralement mené pour réduire la dimension d'un problème.
#
# La construction des composantes principales repose sur l'analyse aux valeurs propres d'une matrice indiquant les niveaux de corrélations entre les caractéristiques. En notant $X$ la matrice contenant nos données qui est donc constituée ici de 4040 lignes et 9 colonnes, la matrice de corrélation des caractéristiques demandée ici est $C = X^TX$.

# %% [markdown]
# ### Construction des composantes principales sur les caractéristiques morphologiques
#
# Construisez une matrice des niveaux de corrélation des caractéristiques. Elle doit être carrée de taille 9x9.

# %%
X = df.to_numpy()
C = np.dot(np.transpose(X), X) #matrice de taille (9,9)

# %% [markdown]
# ### Calcul des vecteur propres et valeurs propres de la matrice de corrélation

# %% [markdown]
# Calculez à l'aide du module `numpy.linalg` les valeurs propres et les vecteurs propres de la matrice $C$. Cette dernière est symétrique définie positive par construction, toutes ses valeurs propres sont strictement positives.

# %%
import numpy.linalg as alg
eigvals, mat = alg.eig(C) #eig renvoie un tuple d'un array valeurs propres de C et d'un array de vecteurs propres en colonnes
eigvects = np.transpose(mat) #pour une meilleure lisibilité

print(eigvals, eigvects) # retourne des arrays contenant les valeurs propres et vecteurs propres de C

# %% [markdown]
# ### Tracé des valeurs propres

# %% [markdown]
# Tracez les différentes valeurs propres calculées en utilisant un axe logarithmique.

# %%
X=np.arange(1, 10, 1)
plt.loglog(X, eigvals, marker='o')
plt.title('Tracé en echelle log-log des valeurs propres')
plt.show()
plt.semilogy(X, eigvals, marker='o')
plt.title('Tracé en echelle semi-log des valeurs propres')
plt.show()

# %% [markdown]
# ### Analyse de l'importance relative des composantes principales

# %% [markdown]
# Vous devriez constater que les valeurs propres décroissent vite. Cette décroissance traduit l'importance relative des composantes principales.
#
# Dans le cas où les valeurs propres sont ordonnées de la plus grande à la plus petite ($\forall (i,j) \in \{1, \ldots, N\}^2, i>j
#  \Rightarrow \lambda_i \leq \lambda_j$), tracez l'évolution de la quantité suivante:
# \begin{equation*}
#  \alpha_i = \frac{\sum\limits_{j=1}^{j=i}\lambda_j}{\sum\limits_{j=1}^{j=N}\lambda_j}
# \end{equation*}
#
# $\alpha_i$ peut être interprété comme *la part d'information du jeu de données initial contenu dans les $i$ premières composantes principales*.

# %%
V = list(eigvals) #listes des valeurs propres déjà triée dans l'ordre décroissant
N = len(V)
LY=[] #liste des alpha_i 
S_N = sum(V[:N])
for i in range (N):
    S_i = sum(eigvals[:i])
    LY.append(S_i / S_N)
Y = np.array(LY)
X = np.arange(0, len(LY), 1)
plt.title(" Tracé en echelle semilogy de l'évolution de alpha_i ")
plt.semilogy(X, Y, marker='+')
plt.show()
plt.title(" Tracé en echelle classique de l'évolution de alpha_i ")
plt.plot(X, Y, marker='+')
plt.show()

# %% [markdown]
# ### Quantité d'information contenue par la plus grande composante principale
#
# Affichez la plus grande valeur propre et le vecteur propre correspondant (ça doit correspondre à la première composante principale).
#
# Quelle est la quantité d'information contenue par cette composante ?
#
# Pratiquement toute l'information ! C'est trop beau pour être vrai non ? 
#
# Affichez les coefficients de cette composante principale. Que remarquez vous ? (*hint* cherchez la caractéristique dont le coefficient est le plus important en valeur absolue) 
#
# En observant les données correspondant à cette caractéristique, avez-vous une idée de ce qui s'est passé ? 

# %%
valpropre_max = V[0] #valeur propre max = la première car liste triée dans l'ordre décroissant
eigenvalue_max = V[0] #valeur propre max = la première car liste triée dans l'ordre décroissant
V2 = list(vectpropres)
vecteurpropre_max = V2[0] #vecteur propre max 
L_vpmax = list(vecteurpropre_max)
print ("La valeur propre max est", V[0], "et le vecteur propre associé est", V2[0]) 


#on sait que l'info est contenue dans les alpha_i
S=sum(V)
print( "La quantité d'information contenue par cette composante est de", V[0]/S, "%.") 

L_eigvectmax = list(vecteurpropre_max)
maxi = max([-min(L_vpmax), max(L_vpmax )]) 
print(maxi) #affiche le coefficient max du vecteur propre
print("Le coefficient max en valeur absolu de cette composante principale est :", maxi) #affiche le coefficient max du vecteur propre

# %% [markdown]
# ## ACP sur les caractéristiques standardisées
#
# #Dans la section précédente, la première composante principale ne prenait en compte que la caractéristique de plus grande variance. Un moyen de s'affranchir de ce problème consiste à **standardiser** les données. Pour un échantillon $Y$ de taille $N$, la variable standardisée correspondante est $Y_{std}=(Y-\bar{Y})/\sigma(Y)$ où $\bar{Y}$ est la moyenne empirique de l'échantillon et $\sigma(Y)$ son écart type empirique. 
#
# #**Notez que** dans notre cas, il faut réaliser la standardisation **caractéristique par caractéristique** (soit colonne par colonne). Si vous n'y avez pas encore pensé, refaites un petit tour sur le cours d'agrégation pour faire ça de manière super efficace ! ;) 
#
# #Menez la même étude que précédement (i.e. à partir de la section `Analyse en composantes principales`) jusqu'à tracer l'évolution des $\alpha_i$.

# %%
import copy 
def standardiser(df):
    df2 = copy.copy(df) 
    for col in df.columns:
        moyenne = np.mean(df[col])
        ecart_type = np.std(df[col])
        for elt in df2[col]:
            elt = (elt - moyenne) / ecart_type
    return(df2)

df2 = standardiser(df)

# %%
X2 = df2.to_numpy()
C2 = np.dot(np.transpose(X2), X2) #matrice de taille (9,9)

eigvals2, mat2 = alg.eig(C2) #eig renvoie un tuple d'un array valeurs propres de C et d'un array de vecteurs propres en colonnes
eigvects2 = np.transpose(mat2) #pour une meilleure lisibilité

X2=np.arange(1, 10, 1)
plt.loglog(X2, eigvals2, marker='o')
plt.title('Tracé en echelle log-log des valeurs propres')
plt.show()
plt.semilogy(X2, eigvals2, marker='o')
plt.title('Tracé en echelle semi-log des valeurs propres')
plt.show()

V2 = list(eigvals2) #listes des valeurs propres déjà triée dans l'ordre décroissant
N2 = len(V2)
LY2=[] #liste des alpha_i 
S_N2 = sum(V2[:N2])
for i in range (N2):
    S_i2 = sum(eigvals[:i])
    LY.append(S_i2 / S_N2)
Y2 = np.array(LY2)
X2 = np.arange(0, len(LY2), 1)
plt.title(" Tracé en echelle semilogy de l'évolution de alpha_i ")
plt.semilogy(X2, Y2, marker='+')
plt.show()
plt.title(" Tracé en echelle classique de l'évolution de alpha_i ")
plt.plot(X2, Y2, marker='+')
plt.show()


# %% [markdown]
# ### Importance des composantes principales
#
# Quelle part d'information est contenue dans les 3 premières composantes principales ? 
#

# %%
print("La quantité d'information contenue par cette composante est de", ((V[0]+V[1]+V[2])/S)*100, "%.")

# %% [markdown]
# Cette part d'information est satisfaisante car nous avons tout de même réduit notre dimension de 9 à 3.

# %% [markdown]
# ### Projection dans la base des composantes principales de nos 4040 défauts
#
# On va convertir les données initiales dans la base des (vecteurs propres des) composantes principales, et les projeter sur l'espace engendré par les premiers vecteurs propres.
#
# Faites attention, vous calculez désormais dans les données standardisées.
#
# Créez une nouvelle dataframe dont les colonnes correspondent aux projections sur le sous-espace des 3 vecteurs propres prépondérants; on appellera ses colonnes P1, P2 et P3

# %%
v1 = eigvects2[0]
v2 = eigvects2[1]
v3 = eigvects2[2]

df_p = pd.DataFrame(index=df2.index, columns=['P1', 'P2', 'P3'])

for i in df_p.index:
    df_p.loc[i, 'P1'] = np.dot(np.array(df2.loc[i]), v1)
    df_p.loc[i, 'P2'] = np.dot(np.array(df2.loc[i]), v2)
    df_p.loc[i, 'P3'] = np.dot(np.array(df2.loc[i]), v3)

# %% [markdown]
# Tracez les nuages de points correspondants dans les plans (P1, P2) et (P1, P3). 

# %%
correlation_plot2(df_p['P1'], df_p['P2'], 
                      xlabel='P1', ylabel='P2',
                      plot_kwargs={'marker': 'x', 'color': 'red'})

correlation_plot2(df_p['P1'], df_p['P3'], 
                      xlabel='P1', ylabel='P2',
                      plot_kwargs={'marker': 'x', 'color': 'red'})

# %% [markdown]
# ## La conclusion
#
# Reprenez maintenant le défaut `4022` et cherchez son plus proche voisin en utilisant la distance euclidienne dans l'espace des composantes principales. Que constatez-vous ? 

# %%
# %matplotlib notebook
from importlib import reload
reload(plt)
from utilities import plot_defect
from importlib import reload

# %%
ligne_4022 = df_p.loc[4022]
s_min = 0
for j in range (3): #on definit la distance euclidienne min initiale comme celle entre le défaut 1 et le 4022 
    s_min += (ligne_4022.iloc[j] - df_p.iloc[0, j]) ** 2
d_min = np.sqrt(s_min)
i_min = 0

for k in range (1, 4040): #on parcourt tous les défauts 
    s = 0 
    if k != 4022 : #on exclu le defaut 4022 ou la distance à lui-meme est nulle 
        for j in range (3):
            s += (ligne_4022.iloc[j] - df_p.iloc[k, j]) ** 2
            distance = np.sqrt(s)
            if distance < d_min :
                d_min = distance
                i_min = k
print(i_min)


# %%
# %matplotlib inline
reload(plt)

# %% [markdown]
# **Une note de fin :** L'objectif de la démarche n'est pas seulement de trouver le plus proche voisin, mais l'ensemble des voisins (recherchez `triangulation de Delaunay` ou `tesselation de Voronoï` si vous être curieux). Or bien que les algorithmes de construction des triangulations/tessellations sont écrits en dimension quelconque, ils sont beaucoup plus efficaces en dimension faible. Dans le cas actuel, une triangulation en dimension 3 est instantanée (avec `scipy.spatial.Delaunay`) alors qu'elle met au moins une heure en dimension 9 (peut-être beaucoup plus, j'ai dû couper car mon train arrivait à destination...)

# %% [markdown]
# ***
