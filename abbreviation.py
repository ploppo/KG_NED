import spacy
from scispacy.linking import EntityLinker
from scispacy.abbreviation import AbbreviationDetector
nlp = spacy.load("en_core_sci_scibert")
# Add the abbreviation pipe to the spacy pipeline.
nlp.add_pipe("abbreviation_detector")
# This line takes a while, because we have to download ~1GB of data
# and load a large JSON file (the knowledge base). Be patient!
# Thankfully it should be faster after the first time you use it, because
# the downloads are cached.
# NOTE: The resolve_abbreviations parameter is optional, and requires that
# the AbbreviationDetector pipe has already been added to the pipeline. Adding
# the AbbreviationDetector pipe and setting resolve_abbreviations to True means
# that linking will only be performed on the long form of abbreviations.
nlp.add_pipe("scispacy_linker", config={"resolve_abbreviations": True, "linker_name": "umls"})
