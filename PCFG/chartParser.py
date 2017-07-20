import argparse
import sys
from nltk import tokenize
from nltk.parse import pchart
from nltk.grammar import Nonterminal, PCFG, ProbabilisticProduction
from nltk.parse.chart import Chart, LeafEdge, TreeEdge, AbstractChartRule

def loadGrammar(args):
	with open(args.grammar_file, 'r') as f:
		pcfg = PCFG.fromstring(f.read())
	return pcfg	
	#prob = getTerminalProbability()

def getProbability(terminal, lang):
	# call the RNN
	return 1

def getTerminalProbability(terminals):
	p = []
	for terminal in terminals: 
		p.append(ProbabilisticProduction(Nonterminal('HI'), [terminal], prob = getProbability(terminal, 'HI')))
		p.append(ProbabilisticProduction(Nonterminal('EN'), [terminal], prob = getProbability(terminal, 'EN')))
	return p	
	
def parseArgument():
	parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    	parser.add_argument('--grammar_file', type=str, default='output.grammar',
                        help='path of the file which stores grammar learned from PCFG')
    	parser.add_argument('--sentence', type=str, default='Ram is a good boy',
                        help='sentence to be parsed')

    	args = parser.parse_args()
	return args

def main(args):
	sentence = args.sentence
	args = parseArgument()
	tokens = sentence.split()
	terminalProductionRules = getTerminalProbability(tokens)
	grammar = loadGrammar(args)
	grammar.productions().extend(terminalProductionRules)
	
	parsers = [
        	#ViterbiParser(grammar),
        	pchart.InsideChartParser(grammar),
        	pchart.RandomChartParser(grammar),
       		pchart.UnsortedChartParser(grammar),
        	pchart.LongestChartParser(grammar),
        	pchart.InsideChartParser(grammar, beam_size = len(tokens)+1)
        ]
	getLength(args, pcfg_grammar, list_nonterm)
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

if __name__=='__main__':
	args = parseArgument()
	main(args)
