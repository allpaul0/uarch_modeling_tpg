mon_projet/
│
├─ data_classes/          # définitions des classes de données
│    ├─ __init__.py
│    ├─ tpg.py            # classe TPG
│    ├─ team.py           # classe Team & CompiledTeam
│    ├─ instruction.py    # classe Instruction
│    ├─ uarch.py          # classe Uarch
│    ├─ measurement.py    # classe TeamMeasurement
│    ├─ features.py       # classe FeatureVector
│    ├─ database.py       # classe DataBase
│    ├─ loader.py         # classe Loader
│
├─ analysis/              # algorithmes de traitement
│    ├─ __init__.py
│    ├─ disassembler.py   # classe Disassembler
│    ├─ analyzer.py       # classe FeaturesAnalyzer
│    ├─ regression.py     # classes Regressor et RegressionModel
│
├─ data/                  # dossiers pour jeux de données (pandas)
│
├─ utils.py               # utilitaires généraux (p. ex. log, file IO)
├─ main.py                # point d’entrée de l’application (script principal)
├─ requirements.txt       # dépendances (pandas, scikit-learn, etc.)
└─ README.md

 python3 main.py inspect --db db_im1.pkl --uarch cv32e40x_im1_zba_zbb |grep -C100 jal