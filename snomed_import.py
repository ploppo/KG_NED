import sys
from tqdm import tqdm

from db_connection import Neo4jConnection
from db_import import BaseImporter

SNOMED_RELS_FILE = 'ontologies/snomed/sct2_Relationship_Full_INT_20230331.txt'
SNOMED_NAMES_FILE = 'ontologies/snomed/sct2_Description_Full-en_INT_20230331.txt'

class SnomedRelationshipsImporter(BaseImporter):
    
    def __init__(self, argv):
        super().__init__(file=SNOMED_RELS_FILE, argv=argv)
    
    def set_constraints(self):
        queries = ["CREATE CONSTRAINT IF NOT EXISTS FOR (n:SnomedEntity) REQUIRE n.id IS UNIQUE",
                   "CREATE INDEX snomedNodeName IF NOT EXISTS FOR (n:SnomedEntity) ON (n.name)",
                   "CREATE INDEX snomedRelationId IF NOT EXISTS FOR ()-[r:SNOMED_RELATION]-() ON (r.id)",
                   "CREATE INDEX snomedRelationType IF NOT EXISTS FOR ()-[r:SNOMED_RELATION]-() ON (r.type)",
                   "CREATE INDEX snomedRelationUmls IF NOT EXISTS FOR ()-[r:SNOMED_RELATION]-() ON (r.umls)"]

        for q in queries:
            self.connection.query(q, db=self.db)
    
    def import_SNOMED_RELS(self):
        query = """
        UNWIND $rows as item
        MERGE (e1:SnomedEntity {id: item.sourceId})
        MERGE (e2:SnomedEntity {id: item.destinationId})
        MERGE (e1)-[:SNOMED_RELATION {id: item.typeId}]->(e2)
        FOREACH(ignoreMe IN CASE WHEN item.typeId = '116680003' THEN [true] ELSE [] END|
            MERGE (e1)-[:SNOMED_IS_A]->(e2)
        )
        """
        size = self.get_file_size()
        data = self.get_rows()
        self.load_in_batch(query, data, size, chunk_size=1000)


class SnomedNamesImporter(BaseImporter):

    def __init__(self, argv):
        super().__init__(file=SNOMED_NAMES_FILE, argv=argv)

    def importNodeNames(self):
        query = """
        UNWIND $rows as item
        MATCH (e:SnomedEntity {id: item.conceptId})
        FOREACH(x in CASE WHEN e.name IS NULL THEN [1] ELSE [] END | 
            SET e.name = item.term
        )
        FOREACH(x in CASE WHEN item.term in e.aliases THEN [] ELSE [1] END | 
            SET e.aliases = coalesce(e.aliases, []) + item.term
        )
        """

        size = self.get_file_size()
        data = self.get_rows()
        self.load_in_batch(query, data, size, chunk_size=1000)
    
    def importRelNames(self):
        query = """
        UNWIND $rows as item
        MATCH (:SnomedEntity)-[r:SNOMED_RELATION {id: item.conceptId}]->(:SnomedEntity)
        FOREACH(x in CASE WHEN r.type IS NULL THEN [1] ELSE [] END | 
            SET r.type = toUpper(replace(item.term, ' ', '_'))
        )
        FOREACH(x in CASE WHEN toUpper(replace(item.term, ' ', '_')) in r.aliases THEN [] ELSE [1] END | 
            SET r.aliases = coalesce(r.aliases, []) + toUpper(replace(item.term, ' ', '_'))
        )
        """

        size = self.get_file_size()
        data = self.get_rows()
        self.load_in_batch(query, data, size, chunk_size=1000)


class SnomedLabelPropagator():
    def __init__(self, argv):
        self.db = argv[3]
        self.connection = Neo4jConnection(uri=argv[0], user=argv[1], pwd=argv[2])
    
    def perform_label_propagation(self):
        root_nodes = self.getSnomedRootNodes()
        self.annotateRootNodes(root_nodes)
        self.propagateLabels(root_nodes)

    def getSnomedRootNodes(self):
        query = """
                MATCH p=(n:SnomedEntity)<-[:SNOMED_IS_A]-(m:SnomedEntity) 
                WHERE n.id= "138875005" // Root node
                RETURN DISTINCT id(m) as id, m.name as label
                """
        
        return self.connection.query(query, db=self.db)

    def annotateRootNodes(self, root_nodes):
        query= """
               MATCH (first_node) WHERE id(first_node)= $id
               WITH first_node, first_node.name as name
               SET first_node:ToBeProcessed
               RETURN first_node
               """
        for i in root_nodes:
            node_id = i['id']
            self.connection.query(query, parameters={'id': node_id}, db=self.db)

    def propagateLabels(self, root_nodes):
        query = """
                MATCH (first_node)
                WHERE id(first_node) = $id
                WITH first_node

                CALL apoc.path.expandConfig(first_node, {
                    relationshipFilter: '<SNOMED_IS_A',
                    minLevel: 1,
                    maxLevel: -1,
                    uniqueness: 'RELATIONSHIP_GLOBAL'
                }) yield path
                UNWIND nodes(path) as other_level
                WITH collect(DISTINCT other_level) as uniques

                UNWIND uniques as unique_other_level
                FOREACH(x in CASE WHEN $label in unique_other_level.type THEN [] ELSE [1] END | 
                SET unique_other_level.type = coalesce(unique_other_level.type, []) + $label
                )
                RETURN unique_other_level
                """
        
        for i in tqdm(root_nodes):
            node_id = i['id']
            node_label = i['label']
            self.connection.query(query, parameters={'id': node_id, 'label': node_label}, db=self.db)

if __name__ == '__main__':
    

    rel_importing = SnomedRelationshipsImporter(argv=sys.argv[1:])
    rel_importing.set_constraints()
    rel_importing.import_SNOMED_RELS()
    
    names_importing = SnomedNamesImporter(argv=sys.argv[1:])
    names_importing.importNodeNames()
    names_importing.importRelNames()

    type_propagator = SnomedLabelPropagator(argv=sys.argv[1:])
    type_propagator.perform_label_propagation()
