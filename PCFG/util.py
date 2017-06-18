from nltk import Tree
import logging
 
 
def corpus2trees(text):
	""" Parse the corpus and return a list of Trees """
	rawparses = text.split("\n\n")
	trees = []
 
	for rp in rawparses:
		if not rp.strip():
			continue
 
		try:
			t = Tree.fromstring(rp)
			trees.append(t)
		except ValueError:
			logging.error('Malformed parse: "%s"' % rp)
 
	return trees
 
 
def trees2productions(trees):
	""" Transform list of Trees to a list of productions """
	productions = []
	for t in trees:
		productions += t.productions()
	return productions
