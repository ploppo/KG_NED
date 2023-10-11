Create a virtual environment:
```virtualenv -p python3 venv```
```source venv/bin/activate```
```pip install neo4j```
```pip install tqdm```
```pip install numpy```

Create a Neo4j database:
```CREATE DATABASE snomed```

Update the input folders in `snomed_import`:
```SNOMED_RELS_FILE = 'ontologies/snomed/sct2_Relationship_Full_US1000124_20220901.txt'```
```SNOMED_NAMES_FILE = 'ontologies/snomed/sct2_Description_Full-en_US1000124_20220901.txt'```

Run the following script:
```giuseppefutia$ python snomed_import.py bolt://localhost:7687 neo4j password snomed```