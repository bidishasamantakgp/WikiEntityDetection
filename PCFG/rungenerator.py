import argparse
from util import getNonterminal
from train import *
from util import *
from generator import *

#def main()
def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data_file', type=str, default='train.txt',
                        help='path of the training file which stores annotated sentences like Penn treebank')
    parser.add_argument('--symbol_file', type=str, default='symbol.txt',
                        help='path of the file which stores non-terminals of the grammar one at each line. The first line should contain the root element')

    parser.add_argument('--num_sentence', type=int, default=10,
                        help='number of sentence to generate')
    args = parser.parse_args()
    pcfg_grammar = generateGrammar(args)
    generate(args, pcfg_grammar)

        
def generate(args, pcfg_grammar):
	#print pcfg_grammar.productions()
        list_nonterm = getNonterminal(args.symbol_file)
        prob_dist = preprocessing(pcfg_grammar, list_nonterm)
        #print prob_dist
	for i in range(args.num_sentence):
        	leafnodes, t, probability = buildtree(pcfg_grammar, prob_dist, list_nonterm[0])
        	#print leafnodes
        	#print len(t)
		print len(t), leafnodes, probability

if __name__=='__main__':
	main()
