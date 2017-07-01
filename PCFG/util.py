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
			t.chomsky_normal_form()
			#trees.append(t.chomsky_normal_form())
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

def getNonterminal(variable_file):
        f = open(variable_file)
        list_nonterm = f.readlines()
        list_nonterm = [l.strip() for l in list_nonterm]
        return list_nonterm
