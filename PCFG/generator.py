import nltk
import sys
import ast
import Queue
from collections import defaultdict
from nltk.grammar import Nonterminal
from nltk.probability import DictionaryProbDist
#from nltk.draw.tree import draw_trees
from util import *

# handle all the probability distribution
def preprocessing(grammar, listnonterm):
	prob_dist = defaultdict(DictionaryProbDist)
	for nonterm in listnonterm:
		prods = grammar.productions(Nonterminal(str(nonterm)))
		dict_rules = {}
		for pr in prods:
			dict_rules[pr.rhs()] = pr.prob()
		prob_dist[nonterm] = DictionaryProbDist(dict_rules)
	return prob_dist


# builds the tree
def buildtree(grammar, prob_dist, start):
	dict_rules = {}
	q = list()
	q.append((start,0))
	leafnodes = []
	terminals = []
	i = 0
	probability = 1
	while(len(q)>0):
		(nonterm, parent) = q.pop(i)
		prods = grammar.productions(Nonterminal(str(nonterm)))
		if(len(prods)>0):
			rule = prob_dist[str(nonterm)].generate()
			if nltk.grammar.is_nonterminal(rule[0]):
                        	probability *= prob_dist[str(nonterm)].prob(rule)
			for el in rule:
				q.append((el,nonterm))
		else:
			leafnodes.append(parent)
			terminals.append(nonterm)	
		
	return (leafnodes,terminals, probability)
