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
    parser.add_argument('--nonterm', type=str, default='SS',
                        help='non terminal whose subtree needs to be generated')
    parser.add_argument('--action', type=str, default='generate',
                        help='generate or getlength')
    parser.add_argument('--length', type=int, default=10,
                        help='generate the combination less than 10 length')

   
    args = parser.parse_args()
    (pcfg_grammar, listnonterm) = generateGrammar(args)
    if args.action == 'generate':
    	generate(args, pcfg_grammar, listnonterm)
    if args.action == 'getlength':
	(lengthlist, leafnode) = getLength(args, pcfg_grammar, listnonterm)
	print lengthlist, leafnode

        
def generate(args, pcfg_grammar, list_nonterm):
        prob_dist = preprocessing(pcfg_grammar, list_nonterm)
	for i in range(args.num_sentence):
        	leafnodes, t, probability = buildtree(pcfg_grammar, prob_dist, args.nonterm)
		print len(t), leafnodes, probability

def getLength(args, pcfg_grammar, list_nonterm):
        prob_dist = preprocessing(pcfg_grammar, list_nonterm)
        length = []
	leafnodeslist = []
	for i in range(args.num_sentence):
                leafnodes, t, probability = buildtree(pcfg_grammar, prob_dist, args.nonterm)
                #print leafnodes 
		if len(t) <= args.length:
			length.append(len(t))
			leafnodeslist.append(leafnodes)
	print leafnodeslist
	return list(set(length)), leafnodeslist

if __name__=='__main__':
	main()
