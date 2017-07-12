import argparse
import importlib
import cPickle
import os
import sys
from nltk import tokenize
from nltk.parse import pchart
from nltk.grammar import Nonterminal, PCFG, ProbabilisticProduction
from nltk.parse.chart import Chart, LeafEdge, TreeEdge, AbstractChartRule
from collections import defaultdict

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

def getProbability(terminals, lang):
        probDict = defaultdict()
	with tf.Session() as sess:
        	tf.global_variables_initializer().run()
        	#saver = tf.train.Saver(tf.global_variables())

		if lang == 'EN':
		    with tf.variable_scope('EN'):
			args.save_dir = args.english_save_dir
			with open(os.path.join(args.english_save_dir, 'config.pkl'), 'rb') as f1:
                        	saved_args = cPickle.load(f1)
                	with open(os.path.join(args.english_save_dir, 'chars_vocab.pkl'), 'rb') as f1:
                        	chars, vocab = cPickle.load(f1)
			#print chars
                	model = char_rnn.Model(saved_args, training=False)
			#model = char_rnn.model.Model(saved_args, training=False)
		if lang == 'HI':
		    with tf.variable_scope('HI'):
			args.save_dir = args.hindi_save_dir
                	with open(os.path.join(args.hindi_save_dir, 'config.pkl'), 'rb') as f1:
                        	saved_args = cPickle.load(f1)
                	with open(os.path.join(args.hindi_save_dir, 'chars_vocab.pkl'), 'rb') as f1:
                        	chars, vocab = cPickle.load(f1)
                	model = char_rnn.Model(saved_args, training=False)
		print chars
		#probDict = defaultdict()
		
		for terminal in terminals:
	    	#with tf.Session() as sess:
			#tf.global_variables_initializer().run()
                	saver = tf.train.Saver(tf.global_variables())
                	ckpt = tf.train.get_checkpoint_state(args.save_dir)
			print ckpt.model_checkpoint_path
                	if ckpt and ckpt.model_checkpoint_path:
                		saver.restore(sess, ckpt.model_checkpoint_path)
                        	#print('DEBUG')
                        	#print(saved_args)
                        	#print(vocab)
                        	#rnnprob *= modeleng.getProbability(sess, eng_chars, eng_vocab, args.sentence, args.n, args.prime,args.sample)[1]
				rnnprob = model.getProbability(sess, chars, vocab, terminal, len(terminal), ' ',0)[1]
                        	probDict[terminal] = rnnprob
		#with tf.Session() as sess:
		#	rnnprob = model.getProbability(sess, chars, vocab, terminal, len(terminal), ' ',0)[1]
		#	probDict[terminal] = rnnprob
		
	print probDict	
	return probDict

def getTerminalProbability(terminals):
	p = []
	enProb = getProbability(terminals, 'EN')
	#hiProb = getProbability(terminals, 'HI')
	for terminal in terminals: 
		#p.append(ProbabilisticProduction(Nonterminal('HI'), [terminal], prob = hiProb[terminal]))
		p.append(ProbabilisticProduction(Nonterminal('EN'), [terminal], prob = enProb[terminal]))
	return p	
	
def parseArgument():
	parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    	parser.add_argument('--grammar_file', type=str, default='output.grammar',
                        help='path of the file which stores grammar learned from PCFG')
    	parser.add_argument('--sentence', type=str, default='Ram is a good boy',
                        help='sentence to be parsed')
	parser.add_argument('--english_save_dir', type=str, default='save',
                        help='model directory to store checkpointed models')
        parser.add_argument('--hindi_save_dir', type=str, default='save',
                        help='model directory to store checkpointed models')

    	args = parser.parse_args()
	return args

def main(args):
	sentence = args.sentence
	args = parseArgument()
	tokens = sentence.split()
	terminalProductionRules = getTerminalProbability(tokens)
	grammar = loadGrammar(args)
	grammar.productions().extend(terminalProductionRules)
	print grammar.productions()
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
        	print('\ns: %s\nparser: %s' % (sentence,parser))
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
if __name__=='__main__':
	args = parseArgument()
	main(args)
