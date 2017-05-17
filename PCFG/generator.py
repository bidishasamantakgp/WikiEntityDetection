import nltk
import sys
import ast
import Queue
from collections import defaultdict
from PCFGParser import *  
from nltk.grammar import Nonterminal
from nltk.probability import DictionaryProbDist
from nltk.draw.tree import draw_trees
from util import corpus2trees, trees2productions
#from queue import *

#viterbi_grammar = PCFGViterbiParser.train(open('smalltrain.txt', 'r'), root='ROOT')
#productions = viterbi_grammar.productions()
def learnpcfg(content, root):
	if not isinstance(content, basestring):
            content = content.read()

        trees = corpus2trees(content)
        productions = trees2productions(trees)
        pcfg = nltk.grammar.induce_pcfg(nltk.grammar.Nonterminal(root), productions)
	#print pcfg.productions()
	return pcfg

def getNonterminal(variable_file):
	f = open(variable_file)
	list_nonterm = f.readlines()
	list_nonterm = [l.strip() for l in list_nonterm]
	return list_nonterm

# handle all the probability distribution
def preprocessing(grammar, listnonterm):
	#dict_rules = {}
	prob_dist = defaultdict(DictionaryProbDist)
	for nonterm in listnonterm:
		print nonterm
		#print grammar.productions(Nonterminal('ROOT'))
		#print grammar.productions(Nonterminal(str(nonterm)))
		prods = grammar.productions(Nonterminal(str(nonterm)))
		#print prods
		dict_rules = {}
		for pr in prods:
			#print pr.rhs()
			dict_rules[pr.rhs()] = pr.prob()
		print dict_rules
		prob_dist[nonterm] = DictionaryProbDist(dict_rules)
	print 'test',prob_dist['NP-SBJ'].generate()		
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
	while(len(q)>0):
		#print i,q
		(nonterm, parent) = q.pop(i)
		#i = i+1
		print nonterm	
		prods = grammar.productions(Nonterminal(str(nonterm)))
		print prods
		if(len(prods)>0):
			#print nonterm, prob_dist[str(nonterm)],prods
			rule = prob_dist[str(nonterm)].generate()
                        print rule
			#for r in rule:
			#	print 'hi',r
			#els = ast.literal_eval(rule)
			for el in rule:
				#q.put((el,nonterm))
				q.append((el,nonterm))
				#i = i+1	
		else:
			leafnodes.append(parent)
			terminals.append(nonterm)	
		
	return (leafnodes,terminals)

if __name__=="__main__":
	# filenames 
	trainfile = sys.argv[1]
	variable_file = sys.argv[2]

	# grammar creation and generation
	viterbi_grammar = learnpcfg(open(trainfile, 'r'), root='ROOT')
	#nonterm = 'ROOT'
	#print nonterm 
	print "##########################"
	print viterbi_grammar.productions(Nonterminal('NP-SBJ'))
	print "##########################"
	#print viterbi_grammar.productions(Nonterminal(nonterm))
	#viterbi_grammar = PCFGViterbiParser.train(open(trainfile, 'r'), root='ROOT')	
	list_nonterm = getNonterminal(variable_file)
	#print list_nonterm
	prob_dist = preprocessing(viterbi_grammar, list_nonterm)
	print prob_dist
	leafnodes, t = buildtree(viterbi_grammar, prob_dist, list_nonterm[0])
	print leafnodes	
	print t
'''	
for pr in s_productions:
	dict_rules[pr.rhs()] = pr.prob()
	s_probDist = DictionaryProbDist(dict_rules
	rule = s_probDist.generate()		
	
	for t in rule:
		deriv = 	
 


t = viterbi_grammar.parse_all(nltk.word_tokenize('Numerous passing references to the phrase have occurred in movies'))
draw_trees(*t)
'''
#print viterbi_parser 
#print t
#print(t.pprint())
#for parse in t:
#	    print "parse"
#           print(parse)

#t.draw()
