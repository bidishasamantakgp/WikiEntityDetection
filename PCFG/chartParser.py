import argparse
import importlib
import cPickle
import pandas as pd
import six
import os
import sys
from nltk import tokenize
from nltk.parse import pchart
from nltk.grammar import Nonterminal, PCFG, ProbabilisticProduction, is_terminal
from nltk.parse.chart import Chart, LeafEdge, TreeEdge, AbstractChartRule
from collections import defaultdict
from rungenerator import *
from collections import defaultdict
import numpy as np

paths = ['../']
for p in paths:
        sys.path.insert(0, p)
import tensorflow as tf

#from char-rnn-tensorflow.model import Model 
char_rnn = importlib.import_module("char-rnn-tensorflow.model")

def loadGrammar(args):
	with open(args.grammar_file, 'r') as f:
		pcfg = PCFG.fromstring(f.read())
	return pcfg	
	#prob = getTerminalProbability()

def getModel(args, lang, segments=[]):
 	if lang == 'en':
		print "Inside en"
                with open(os.path.join(args.english_save_dir, 'config.pkl'), 'rb') as f1:
			saved_args = cPickle.load(f1)
                with open(os.path.join(args.english_save_dir, 'chars_vocab.pkl'), 'rb') as f1:
			chars, vocab = cPickle.load(f1)
		with tf.variable_scope('en'):
			model = char_rnn.Model(saved_args, training=False)
		return getProbability(segments,model,chars, vocab,'en', args)
        if lang == 'hi':
		print "inside hi"
		with open(os.path.join(args.hindi_save_dir, 'config.pkl'), 'rb') as f1:
			saved_args = cPickle.load(f1)
		with open(os.path.join(args.hindi_save_dir, 'chars_vocab.pkl'), 'rb') as f1:
			chars, vocab = cPickle.load(f1)
		with tf.variable_scope('hi'):
                	model = char_rnn.Model(saved_args, training=False)
		return getProbability(segments,model,chars, vocab, 'hi',args)

	#return (model, chars, vocab)

def getProbability(segments, model, chars, vocab, lang, args):
        print 'Lang', lang
	probDict = defaultdict()
	with tf.Session() as sess:
        	tf.global_variables_initializer().run()
                saver = tf.train.Saver([v for v in tf.all_variables() if lang in v.name])
		#saver = tf.train.Saver(tf.global_variables())
                ckpt = tf.train.get_checkpoint_state(args.save_dir)
		#print ckpt.model_checkpoint_path
                for segment in segments:
			#print segment
			if ckpt and ckpt.model_checkpoint_path:
                		saver.restore(sess, ckpt.model_checkpoint_path)
				rnnprob = model.getProbability(sess, chars, vocab, segment, len(segment), '',0)[1]
				#print rnnprob	
			probDict[segment] = rnnprob
	return probDict

def createSegment(length, sentence):
	tokens = sentence.split()
	i = 0
	segments = []
	while i < len(tokens)-length + 1:
		segments.append(' '.join(tokens[i:i+length]))
		i = i + 1
	#print "Segments", segments
	
	return segments

def getTerminalProbability(args, pcfg_grammar, list_nonterm):
	#args.save_dir = args.english_save_dir
	#(modelen, charsen, vocaben) = getModel(args, 'en')
	#args.save_dir = args.hindi_save_dir
	#(modelhi, charhi, vocabhi) = getModel(args, 'hi')
	
	p = []
	args.nonterm = 'HS'
	args.save_dir = args.hindi_save_dir
	args.num_sentence = 1000
	args.length = len(args.sentence)
	segmentshi = []
	#print "PCFG grammar", pcfg_grammar	
	(lengthlist, listterminal) = getLength(args, pcfg_grammar, list_nonterm)
	#print "lengthlist", lengthlist
	for length in lengthlist:
		segmentshi.extend(createSegment(length, args.sentence.lower()))
	probdicthi = getModel(args, 'hi', segmentshi)
	listProb = probdicthi.values()
	segmentshi = list(set(segmentshi))
	
	args.nonterm = 'ES'
        args.save_dir = args.english_save_dir
	(lengthlist, listterminal) = getLength(args, pcfg_grammar, list_nonterm)
	segmentsen = []
        for length in lengthlist:
                segmentsen.extend(createSegment(length, args.sentence.lower()))
        probdicten = getModel(args, 'en', segmentsen)
        listProb.extend(probdicten.values())
	segmentsen = list(set(segmentsen))
	
	listProb = sorted(listProb)
	denom = (len(listProb) * (len(listProb)+1)) / 2
	prob1 = 0
	for segment in segmentshi:
		probnew = (listProb.index(probdicthi[segment])+ 1.0)/(denom+1)
		probnew = float("{0:.8f}".format(round(probnew,8)))
		prob1 += probnew
		#print segment, probnew
		p.append(ProbabilisticProduction(Nonterminal('HS'), [Nonterminal(token.upper()) for token in segment.split()], prob = probnew))
	p.append(ProbabilisticProduction(Nonterminal('HS'), ['Dummy'], prob = (1.0 - prob1) ))
	
	#print 'HS', prob1, 1.0-prob1
	prob1 = 0
	for segment in segmentsen:
		probnew = (listProb.index(probdicten[segment])+ 1.0)/(denom+1)
		probnew = float("{0:.8f}".format(round(probnew,8)))
                prob1 += probnew
		p.append(ProbabilisticProduction(Nonterminal('ES'), [Nonterminal(token.upper()) for token in segment.split()], prob = probnew))
	
	#print 'ES',prob1, 1.0-prob1
	p.append(ProbabilisticProduction(Nonterminal('ES'), ['Dummy'], prob = (1.0 - prob1) ))

	return p
	
def parseArgument():
	parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    	parser.add_argument('--grammar_file', type=str, default='output.grammar',
                        help='path of the file which stores grammar learned from PCFG')
    	parser.add_argument('--sentence', type=str, default='',
                        help='sentence to be parsed')
	parser.add_argument('--english_save_dir', type=str, default='save',
                        help='model directory to store checkpointed models')
        parser.add_argument('--hindi_save_dir', type=str, default='save',
                        help='model directory to store checkpointed models')

    	args = parser.parse_args()
	return args

def getnonterm(grammar):
	nontermlist = []
	for p in grammar.productions():
		nontermlist.append(str(p.lhs()))
	return nontermlist

def main(args):
	
	sentence = args.sentence.lower()
	args.sentence = sentence
	tokens = sentence.split()
	grammar = loadGrammar(args)
	nonterm = getnonterm(grammar)
	terminalProductionRules = getTerminalProbability(args, grammar, nonterm)
	HSrules = grammar.productions(Nonterminal('HS'))
	for rule in HSrules:
		grammar.productions().remove(rule)
	
	ESrules = grammar.productions(Nonterminal('ES'))
        for rule in ESrules:
                grammar.productions().remove(rule)

	grammar.productions().extend(terminalProductionRules)
	
	for token in tokens:
		grammar.productions().append(ProbabilisticProduction(Nonterminal(token.upper()), [unicode(token)], prob = 1))

	#print "Grammars"
        grammarlist = str(grammar).split('\n')[1:]
	
	#print "Transfered"
	strgrammar = ''
	for p in grammar.productions():
		rhs  = p.rhs()
		rhsstr=''
		for r in rhs:
			if is_terminal(r):
				rhsstr += '\''+str(r)+'\' '
			else:
				rhsstr += str(r)+' '
                strgrammar += str(p.lhs())+ ' -> '+ rhsstr +' ['+ '{0:.8f}'.format(p.prob())+']\n'
	#print strgrammar
	
	grammar = PCFG.fromstring(strgrammar.split('\n'))
	#'''
	#grammar = loadGrammar(args)

	#tokens = args.sentence.lower().split()
        #nonterm = getnonterm(grammar)

	CYK(tokens, nonterm, grammar)
	#with open(args.grammar_file, 'r') as f:
        #        content = f.read()

	
	#trees = corpus2trees(content)
        #productions = trees2productions(trees)
        #listnonterm = []
        #grammar = nltk.grammar.induce_pcfg(nltk.grammar.Nonterminal('SS'), productions)
	#print grammar
	
	#'''
	'''
	parsers = [
        	#ViterbiParser(grammar),
        	pchart.InsideChartParser(grammar),
        	pchart.RandomChartParser(grammar),
       		pchart.UnsortedChartParser(grammar),
        	pchart.LongestChartParser(grammar),
        	pchart.InsideChartParser(grammar, beam_size = len(tokens)+1)
        ]
	# Run the parsers on the tokenized sentence.
    	times = []
    	average_p = []
    	num_parses = []
    	all_parses = {}
    	for parser in parsers:
        	print('\ns: %s\nparser: %s' % (args.sentence,parser))
       		#parser.trace(3)
        	#t = time.time()
        	print "inside parsers"
		parses = list(parser.parse(tokens))
        	for p in parses:
			print p
		#break
		#times.append(time.time()-t)
        	#p = (reduce(lambda a,b:a+b.prob(), parses, 0)/len(parses) if parses else 0)
        	#average_p.append(p)
        	#num_parses.append(len(parses))
        	#for p in parses: all_parses[p.freeze()] = 1
	'''
def preprocessingGrammar(grammar):
	parent = defaultdict(list)
	probdict = {}
	for prod in grammar.productions():
		if is_terminal(prod.rhs()[0]):
			#print prod.rhs()[0]
			parent[str(prod.rhs()[0])].append(str(prod.lhs()))
		probdict[str(prod.lhs())+' -> '+' '.join([str(x) for x in prod.rhs() if len(str(x))>0])] = prod.prob()
	return (parent, probdict)


def CYK(tokens, listnonterm, grammar):
	listnonterm = list(set(listnonterm))
	#print listnonterm
	n = len(tokens)
	#no_nonterm = len(listnonterm)
	HIrules = grammar.productions(Nonterminal('HI'))
        HIrules.extend(grammar.productions(Nonterminal('EN')))
        #print HIrules
        for rule in HIrules:
                grammar.productions().remove(rule)
	for nonterm in listnonterm:
		prods = grammar.productions(Nonterminal(nonterm))
		flag = True
		for prod in prods:
			if 'HS' in prod.rhs() or 'ES' in prod.rhs() or str(prod.lhs()) in [str(x.upper()) for x in tokens] or str(prod.lhs()) in ['SS', 'HS', 'ES', 'START']:
				flag = False
		if flag==True:
			listnonterm.remove(nonterm)
			for prod in prods:
				if prod in grammar.productions():
					grammar.productions().remove(prod)
	#print listnonterm
	no_nonterm = len(listnonterm)

	(parent, probdict) = preprocessingGrammar(grammar)
	#print parent
	#print probdict

	PI = np.zeros((n,n,no_nonterm))
	BP = np.zeros((n,n,no_nonterm), dtype=(int,3))
	BPAUX = defaultdict(lambda: defaultdict(list))
	for i in range(n):
		for j in range(no_nonterm):
			if str(listnonterm[j]) in parent[tokens[i]]:
				PI[i][i][j] = probdict[listnonterm[j]+' -> '+tokens[i]]
				#print i,j, PI[i][i][j], listnonterm[j],  probdict[listnonterm[j]+' -> '+tokens[i]]
	#print PI
	## logic for HS and ES will come here
	
	#for i in range(n):
	#	print i,PI[i][i]
	for l in range(1,n):
		for i in range(n-l): 
			j = i + l
			for x in range(no_nonterm):
				productions = grammar.productions(Nonterminal(listnonterm[x]))
				for prod in productions:
					if len(prod.rhs())==1:
						#print 'continuing', prod
						continue
					key= str(prod.lhs())+' -> '+' '.join([x1.upper() for x1 in tokens[i:j+1]])
                        		#if probdict.has_key(str(prod.lhs())+' -> '+' '.join([x.upper() for x in tokens[i:j+1]])):
                        		if key in probdict.keys():
                                		#print 'Key exists'
                                		if (PI[i][j][x] < probdict[str(prod.lhs())+' -> '+' '.join([x1.upper() for x1 in tokens[i:j+1]])]):
                                        		 PI[i][j][x] = probdict[str(prod.lhs())+' -> '+' '.join([x1.upper() for x1 in tokens[i:j+1]])]
							 BPAUX[i][j].append([listnonterm[x],s,'','',PI[i][j][x]])
							 continue

					y = str(prod.rhs()[0])
					z = str(prod.rhs()[1])
					if y in listnonterm and z in listnonterm:
					   for s in range(i,j):
						key = str(prod.lhs())+' -> '+y+' '+z
						#print key
						if key in probdict.keys():
							if (PI[i][j][x]< probdict[str(prod.lhs())+' -> '+y+' '+z] * PI[i][s][listnonterm.index(y)] * PI[s+1][j][listnonterm.index(z)]):
								#print "inside"
								PI[i][j][x] = probdict[str(prod.lhs())+' -> '+y+' '+z] * PI[i][s][listnonterm.index(y)] * PI[s+1][j][listnonterm.index(z)]
								BP[i][j][x] = (s,listnonterm.index(y),listnonterm.index(z))
								BPAUX[i][j].append([listnonterm[x],s,y,z,PI[i][j][x]])
	np_array = PI.tolist()
	df = pd.DataFrame(np_array)	
	df.to_csv("ex2.2.csv")
	df1 = pd.DataFrame(BP.tolist())
	df1.to_csv("ex2_BP.2.csv")
	df2 = pd.DataFrame(BPAUX)
	df2.to_csv("ex2_BPAUX1.2.csv")
	parents = [x[0] for x in BPAUX[0][len(tokens)-1]]
	for p in parents:
		build_trees(0, len(tokens)-1, tokens, 0, p, BPAUX, PI, listnonterm)

def dfs(BPAUX,node):
    global visited
    if node not in visited:
        visited.append(node)
        for n in graph[node]:
            dfs(graph,n)

def build_trees(s, e, tokens,spacecount,parent,BPAUX, PI, listnonterm):
	#print s,e
	for x in BPAUX[s][e]:
		# BPAUX[s][e]
		if x[0] == parent:	
			print ' '.join('\t' for n in range(spacecount)), x[0],'->',x[2],' ', x[3], tokens[s:e+1], PI[s][e][listnonterm.index(x[0])]
			mid = x[1]
			build_trees(s, mid, tokens, spacecount + 1, x[2], BPAUX, PI, listnonterm) 
			build_trees(mid+1, e, tokens, spacecount + 1, x[3], BPAUX, PI, listnonterm)
	

if __name__=='__main__':
	args = parseArgument()
	main(args)
