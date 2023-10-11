### CANDIDATE SELECTION
from db_import import BaseImporter2


class full_text(BaseImporter2):

    def __init__(self, argv):
        super().__init__(argv=argv)

    def search(self, text):
        q = 'CALL db.index.fulltext.queryNodes("prova","'+str(text)+'") YIELD node, score RETURN node.name, node.id, score'
        a = self.connection.query(q, db=self.db)

        return a
