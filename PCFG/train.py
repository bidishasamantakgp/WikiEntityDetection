import argparse
from util import *
import nltk
from nltk.grammar import Production, Nonterminal, ProbabilisticProduction

#from PCFGParser import *

def learnpcfg(content, root):
        if not isinstance(content, basestring):
            content = content.read()

        trees = corpus2trees(content)
        productions = trees2productions(trees)
        pcfg = nltk.grammar.induce_pcfg(nltk.grammar.Nonterminal(root), productions)
        #print pcfg.productions()
        return pcfg

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data_file', type=str, default='train.txt',
                        help='path of the training file which stores annotated sentences like Penn treebank')
    parser.add_argument('--symbol_file', type=str, default='symbol.txt',
                        help='path of the file which stores non-terminals of the grammar one at each line. The first line should contain the root element')

    args = parser.parse_args()
    print generateGrammar(args)

	

def generateGrammar(args):
     trainfile = args.data_file
     variablefile = args.symbol_file
     list_nonterm = getNonterminal(variablefile)
     pcfg_grammar = learnpcfg(open(trainfile, 'r'), root=list_nonterm[0])
     return pcfg_grammar

if __name__=="__main__":
     main()
