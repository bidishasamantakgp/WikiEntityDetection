import nltk
import sys
import ast
import Queue
from collections import defaultdict
from nltk.grammar import Nonterminal
from nltk.probability import DictionaryProbDist
from nltk.draw.tree import draw_trees
from util import *

# handle all the probability distribution
def preprocessing(grammar, listnonterm):
	#dict_rules = {}
	prob_dist = defaultdict(DictionaryProbDist)
	for nonterm in listnonterm:
		#print nonterm
		#print grammar.productions(Nonterminal('ROOT'))
		#print grammar.productions(Nonterminal(str(nonterm)))
		prods = grammar.productions(Nonterminal(str(nonterm)))
		#print prods
		dict_rules = {}
		for pr in prods:
			#print pr.rhs()
			dict_rules[pr.rhs()] = pr.prob()
		#print dict_rules
		prob_dist[nonterm] = DictionaryProbDist(dict_rules)
	#print 'test',prob_dist['NP-SBJ'].generate()		
	return prob_dist


# builds the tree
def buildtree(grammar, prob_dist, start):
	dict_rules = {}
	#q = Queue()
	q = list()
	#q.put(start)
	q.append((start,0))
	leafnodes = []
	terminals = []
	#while(not q.empty()):
	i = 0
	probability = 1
	while(len(q)>0):
		#print i,q
		(nonterm, parent) = q.pop(i)
		#i = i+1
		#print nonterm	
		prods = grammar.productions(Nonterminal(str(nonterm)))
		#print prods
		if(len(prods)>0):
			#print nonterm, prob_dist[str(nonterm)],prods
			rule = prob_dist[str(nonterm)].generate()
			if nltk.grammar.is_nonterminal(rule[0]):
                        	probability *= prob_dist[str(nonterm)].prob(rule)
			for el in rule:
				q.append((el,nonterm))
		else:
			leafnodes.append(parent)
			terminals.append(nonterm)	
		
	return (leafnodes,terminals, probability)
