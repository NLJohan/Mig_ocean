import numpy as np
import matplotlib.pyplot as plt
import random as rd
import scipy.signal


def echantillonage(image_s,GSD_s,d=0.14):
    """
    Entrées : 
    - image_s : np.array : tableau numpy carré de taille n représentant la bande s après convolution avec la PSF de la bande s
    - GSD_s   : float    : GSD de la bande s (0.7m pour le PAN et 2.8m pour R,G,B,MIR)
    - d=0.14  : float    : constante de longueur (en m) de PLEIADES telle que d divise les GSD de toutes les bandes ET telle que d est plus petite que le pouvoir de résolution minimal de PLEIADES cette constante est fixée à d=0.14m < GSD_min = 0.26m et qui divise 0.7m (résolution du pan) et 2.8m (résolution des R,G,B,MIR)

    Sorties :
    - jitter  : np.array : tableau numpy carré de taille (n*d/GSD) représentant la bande s de l'image échantillonnée d'après le jitter
    - p       : int      : entier aléatoire compris dans [0,(GSD_s/d)²-1] qui représente le jitter choisi

    Description :
    echantillonage choisit aléatoirement un pixel p parmi (GSD_s**2/d**2) possibles et crée une nouvelle image suivant le procédé du jitter.
    pour chaque groupe carré de taille (GSD/d) composant l'image, le p-ième pixel est extrait et la valeur de chaque pixel du groupe est passée à celle du pixel p
    """
    n,l,_=np.shape(image_s)
    if not int(GSD_s/d)%n==0 and int(GSD_s/d)%l==0:
        print("ERROR : jitter does not divide shape of image")
        return None
    else :
        p=rd.randrange(GSD_s**2/d**2)
        jitter=np.empty(shape=(int(n/GSD_s),int(p/GSD_s),5))
        for i in range(int(n/GSD_s)):
            for k in range(int(l/GSD_s)):
                jitter[i,k,:]=image_s[i*GSD_s+p//GSD_s,k*GSD_s+p%GSD_s,:]
        return jitter,p



def convolution_psf(psf_s,image_s):
    """
    Entrées : psf_s   : np.array : tableau de taille 512 qui représente la PSF de la bande s
              image_s : np.array : tableau numpy de taille n+511 qui représente la photo initiale de la bande s

    Sortie  : _       : np.array : tableau numpy de taille n qui représente l'image convoluée de la bande s

    Description :
    convolution_psf renvoie l'image convoluée par la PSF 
    """
    return scipy.signal.fftconvolve(image_s,psf_s)[255:-256,255:-256]

