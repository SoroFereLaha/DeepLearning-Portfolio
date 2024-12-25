Pour utiliser ce code :

1. Créez une structure de dossiers comme suit :
```
base_path/
├── dataset_1/
│   ├── jokes/
│   │   └── init1.html à init100.html
│   └── ratings_1.xls, ratings_2.xls, ratings_3.xls
├── dataset_3/
│   ├── jokes.xls
│   └── ratings.xls
└── dataset_4/
    ├── jokes.xls
    └── ratings.xls
```

2. Modifiez le `base_path` dans le code pour pointer vers votre dossier racine

3. Exécutez le script

Le script va :
- Traiter tous les datasets Jester
- Gérer les différents formats (HTML et Excel)
- Marquer les blagues retirées et le "gauge set"
- Combiner toutes les données
- Générer des statistiques détaillées

Les particularités prises en compte :
- Les 22 blagues retirées
- Le "gauge set" de 8 blagues
- Les valeurs 99 pour les évaluations manquantes
- Les différents formats entre les datasets
- La cohérence des IDs de blagues entre les datasets
