import nltk
from nltk.grammar import Production, Nonterminal, ProbabilisticProduction
from util import corpus2trees, trees2productions
 
 
class PCFGViterbiParser(nltk.PCFG):
    def __init__(self, grammar, trace=0):
        super(PCFGViterbiParser, self).__init__(grammar, trace)
 
    @staticmethod
    def _preprocess(tokens):
        replacements = {
            "(": "-LBR-",
            ")": "-RBR-",
        }
        for idx, tok in enumerate(tokens):
            if tok in replacements:
                tokens[idx] = replacements[tok]
 
        return tokens
 
    @classmethod
    def train(cls, content, root):
        if not isinstance(content, basestring):
            content = content.read()
 
        trees = corpus2trees(content)
        productions = trees2productions(trees)
        pcfg = nltk.grammar.induce_pcfg(nltk.grammar.Nonterminal(root), productions)
        #print(pcfg)
	#print(pcfg.productions())
	return cls(pcfg)
 
    def parse(self, tokens):
        tokens = self._preprocess(list(tokens))
        tagged = nltk.pos_tag(tokens)
 
        missing = False
        for tok, pos in tagged:
            if not self._grammar._lexical_index.get(tok):
                missing = True
                self._grammar._productions.append(ProbabilisticProduction(Nonterminal(pos), [tok], prob=0.000001))
        if missing:
            self._grammar._calculate_indexes()
 
        print 'HI'
	testlist = super(PCFGViterbiParser, self).parse(tokens)
	for test in testlist:
		test.draw()
	return super(PCFGViterbiParser, self).parse(tokens)
