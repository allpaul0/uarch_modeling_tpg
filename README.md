mon_projet/
│
├─ data_classes/          # définitions des classes de données
│    ├─ __init__.py
│    ├─ tpg.py            # classe TPG
│    ├─ team.py           # classe Team et Instruction
│    ├─ uarch.py          # classe Uarch
│    ├─ measurement.py    # classe TeamMeasurement
│    ├─ features.py       # classe FeatureVector
│
├─ analysis/               # algorithmes de traitement
│    ├─ __init__.py
│    ├─ disassembler.py    # classe Disassembler
│    ├─ analyzer.py       # classe FeaturesAnalyzer
│    ├─ regression.py     # classes Regressor et RegressionModel
│
├─ data/                  # dossiers pour jeux de données (pandas)
│
├─ utils.py               # utilitaires généraux (p. ex. log, file IO)
├─ main.py                # point d’entrée de l’application (script principal)
├─ requirements.txt       # dépendances (pandas, scikit-learn, etc.)
└─ README.md
