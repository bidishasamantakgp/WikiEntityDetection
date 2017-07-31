import importlib
import argparse
import cPickle
import os
import ast
import sys
import csv
from collections import defaultdict
from math import log
from translatetransliterate_util import translate, transliterate

paths = ['../']
for p in paths:
        sys.path.insert(0, p)
import tensorflow as tf
char_rnn = importlib.import_module("char-rnn-tensorflow.model")

def main():

	parser = argparse.ArgumentParser(
                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        parser.add_argument('--data_file', type=str, default='cricket.txt',
                        help='data file to store the corpus which needs to be transformed to code mixed')
        parser.add_argument('--segment_file', type=str, default='segment.txt',
                        help='file to store the segmentation information genearted by a grammar')

	parser.add_argument('--source_lang', type=str, default='en',
                        help='source language hi, en')
	parser.add_argument('--matric_lang', type=str, default='English',
                        help='source language hi, en')
        parser.add_argument('--target_lang', type=str, default='hi',
                        help='target language to be converted en, hi')

	parser.add_argument('--save_dir', type=str, default='save',
                        help='model directory to store checkpointed models')
	parser.add_argument('--output_csv', type=str, default='output.csv',
                        help='csvfile to store the output')

	parser.add_argument('--english_save_dir', type=str, default='save',
                        help='model directory to store checkpointed models')
	parser.add_argument('--hindi_save_dir', type=str, default='save',
                        help='model directory to store checkpointed models')
    	#parser.add_argument('-n', type=int, default=500,
        #                help='number of characters to sample')
    	parser.add_argument('--prime', type=str, default=' ',
                        help='first word of the sentence')
    	parser.add_argument('--sample', type=int, default=1,
                        help='0 to use max at each timestep, 1 to sample at '
                             'each timestep, 2 to sample on spaces')
    	#parser.add_argument('--sentence', type=str, default='hello world',
        #                help='put your sentence fragment to get the probability value')


	args = parser.parse_args()
	getcodemixswitch(args)

	#dict_rules = builddictionary(args)
	#print dict_rules[10]
	#parsecorpora(args, dict_rules)

def builddictionary(args):
	f = open(args.segment_file, 'r')
	dict_rules = defaultdict(list)
	for line in f:
		#print line
		line = line.strip()
		tokens = line.split('\t')
		#print tokens[1], tokens[2]
		temp_list = ast.literal_eval(tokens[1])
		temp_list.append(tokens[2])
		#print temp_list
		dict_rules[int(tokens[0])].append(list(temp_list))
		#ast.literal_eval(tokens[1]).append(tokens[2])))
		#dict_rules[int(tokens[0])].append(tokens[2])
	return dict_rules

def convertsegment(args, segments):
	args.save_dir = args.hindi_save_dir
	transliterated = []
	for segment in segments:
        	args.data = segment
		translated = translate(args)
        	args.data = translated[0]
        	if len(args.data) == 0:
                	transliterated.append(' ')
			continue
		text = transliterate(args)
		if len(text) == 0:
			transliterated.append(' ')
		else:
        		transliterated.append(text[0])
	return transliterated


def getsegments(tags, sentence):
	tokens = sentence.split()
	hiseg = ''
	enseg = ''
	enseglist = []
	segments = []
	hiseglist = []
	
		
	for i in range(len(tags)-1):
		if tags[i] != 'EN':
			if i>0 and tags[i-1] == 'EN':
				enseglist.append(enseg.strip())
				segments.append((enseg.strip(), 'en'))
				enseg = ''
			hiseg+=' '+tokens[i]
		if tags[i] == 'EN':
			if i>0 and tags[i-1] != 'EN':
				hiseglist.append(hiseg.strip())
				segments.append((hiseg.strip(), 'other'))
				hiseg = ''
			enseg+=tokens[i] + ' '
	
	
	if tags[-2] != 'EN':
		segments.append((hiseg.strip(), 'other'))
		hiseglist.append(hiseg.strip())
	else:
		segments.append((enseg.strip(), 'en'))
                enseglist.append(enseg.strip())
	return (segments, hiseglist, enseglist)

def getProbability_help(segments, model, chars, vocab, lang, args):
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

def getcodemixswitch(args):
	dict_rules = builddictionary(args) 
	f = open(args.data_file, 'r')
	#lines = f.readlines()
	to_be_translated = []
	sentences = []
	eng_list = []
	problist = []
	to_be_translated = []
	for line in f:
		line = line.lower()
		tokens = line.split()
		sen_length = len(tokens)
		if sen_length in dict_rules.keys():
			tags_list = dict_rules[sen_length]
			for tags in tags_list:
				print tags
				(segments, otherseg, enseg) = getsegments(tags, line)
				print otherseg	
				to_be_translated.extend(otherseg)
				eng_list.extend(enseg)
				sentences.append(segments)
				problist.append(tags[-1])
				#break
				#otherlangseg.append(otherseg)
		#break
	translated =  convertsegment(args, to_be_translated)
        args.save_dir = args.hindi_save_dir	
	probdictother = getModel(args, translated, args.target_lang)
	args.save_dir = args.english_save_dir
	probdicten = getModel(args,  eng_list, 'en')
	
	i = 0
	with open(args.output_csv, 'a') as csvfile:
             fieldnames = ['original sentence', 'codemixed', 'segment probability eng','segment probability hi', 'switching probability', 'totoal']
             writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
	     writer.writeheader()
	
	for sentence in sentences:
		codeswitched = ''
		probeng = 0
		probhi = 0
		for (segment, lang) in sentence: 
			segment = segment.strip()
			if lang == 'en':
				probeng +=  log(probdicten[segment])
				codeswitched += ' '+segment
			else:
				#print segment
				probhi += log(probdictother[translated[to_be_translated.index(segment)]])
				codeswitched += ' '+translated[to_be_translated.index(segment)]
		print sentence, codeswitched, probhi, probeng, problist[i]
		sent = ' '.join([x[0] for x in sentence])
		with open(args.output_csv, 'a') as csvfile:
			writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    			writer.writerow({fieldnames[0]: sent, fieldnames[1]: codeswitched, fieldnames[2]: str(probeng), fieldnames[3]: str(probhi), fieldnames[4]: str(log(float(problist[i]))), fieldnames[5]: str(probeng+probhi+log(float(problist[i])))})
		i = i + 1

def getModel(args, segments, lang):
        if lang == 'en':
                print "Inside en"
                with open(os.path.join(args.english_save_dir, 'config.pkl'), 'rb') as f1:
                        saved_args = cPickle.load(f1)
                with open(os.path.join(args.english_save_dir, 'chars_vocab.pkl'), 'rb') as f1:
                        chars, vocab = cPickle.load(f1)
                with tf.variable_scope('en'):
                        model = char_rnn.Model(saved_args, training=False)
                return getProbability_help(segments,model,chars, vocab,'en', args)
        if lang == 'hi':
                print "inside hi"
                with open(os.path.join(args.hindi_save_dir, 'config.pkl'), 'rb') as f1:
                        saved_args = cPickle.load(f1)
                with open(os.path.join(args.hindi_save_dir, 'chars_vocab.pkl'), 'rb') as f1:
                        chars, vocab = cPickle.load(f1)
                with tf.variable_scope('hi'):
                        model = char_rnn.Model(saved_args, training=False)
                return getProbability_help(segments,model,chars, vocab, 'hi',args)


if __name__=='__main__':
	
	main()
