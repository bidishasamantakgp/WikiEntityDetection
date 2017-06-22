import importlib
import argparse
import cPickle
import os
import ast
import sys
from collections import defaultdict
from translatetransliterate_util import translate, transliterate
paths = ['../']
for p in paths:
        sys.path.insert(0, p)
import tensorflow as tf
char_rnn = importlib.import_module("char-rnn-tensorflow")
#__import__('../char-rnn-tensorlfow')

#from char_rnn_tensorflow import calculateprob as calc
#from char_rnn_tensorflow.model import Model as model

#from rnncode.modeleng import Model as engModel
#import tensorflow as tf

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
	dict_rules = builddictionary(args)
	#print dict_rules[10]
	parsecorpora(args, dict_rules)

def builddictionary(args):
	f = open(args.segment_file, 'r')
	dict_rules = defaultdict(list)
	for line in f:
		tokens = line.split('\t')
		#print tokens[1], tokens[2]
		temp_list = ast.literal_eval(tokens[1])
		temp_list.append(tokens[2])
		#print temp_list
		dict_rules[int(tokens[0])].append(list(temp_list))
		#ast.literal_eval(tokens[1]).append(tokens[2])))
		#dict_rules[int(tokens[0])].append(tokens[2])
	return dict_rules

def parsecorpora(args, dict_rules):
	f = open(args.data_file, 'r')
        if args.matric_lang == 'Hindi':
		with open(os.path.join(args.english_save_dir, 'config.pkl'), 'rb') as f1:
        		eng_saved_args = cPickle.load(f1)
    		with open(os.path.join(args.english_save_dir, 'chars_vocab.pkl'), 'rb') as f1:
        		eng_chars, eng_vocab = cPickle.load(f1)
    		modeleng = char_rnn.model.Model(eng_saved_args, training=False)

	else:
		with open(os.path.join(args.hindi_save_dir, 'config.pkl'), 'rb') as f1:
                	hindi_saved_args = cPickle.load(f1)
        	with open(os.path.join(args.hindi_save_dir, 'chars_vocab.pkl'), 'rb') as f1:
                	hindi_chars, hindi_vocab = cPickle.load(f1)

		modelhindi = char_rnn.model.Model(hindi_saved_args, training=False)
	
	for line in f:
		#englishsegment = ''
		#hindisegment = ''
		tokens = line.split()
		sen_length = len(tokens)
		print sen_length
		if sen_length in dict_rules.keys():
		   segments_list = dict_rules[sen_length]
		   englishsegment = ''
                   hindisegment = ''
		   for segments in segments_list:
			englishsegment = ''
                   	hindisegment = ''
			print 'test',segments
			#print line
			rnnprob = 1
			changedsen = ''
			#segments = dict_rules[sen_length]
			prob = float(segments[-1])
			for i in range(0,len(segments)-1):
				if segments[i] != 'HI':
					segments[i] = 'EN'
				#print segments[i], 'HI', hindisegment, 'EN', englishsegment, changedsen
				if i != 0 and segments[i-1] != segments[i]:
					#if segments[i-1] == 'EN' or segments[i-1] == 'HASH':
					if segments[i] == 'HI':
						#print 'English segment', englishsegment
						#args.data = englishsegment
					     hindisegment = tokens[i]+ ' '
					     if args.matric_lang == 'English':
					     	changedsen += ' ' + englishsegment
					     if args.matric_lang == 'Hindi':	
						args.save_dir = args.english_save_dir
                                                args.sentence = englishsegment
						args.n = len(englishsegment)
						
                                                with tf.Session() as sess:
                                                        tf.global_variables_initializer().run()
                                                        saver = tf.train.Saver(tf.global_variables())
                                                        ckpt = tf.train.get_checkpoint_state(args.save_dir)
                                                        if ckpt and ckpt.model_checkpoint_path:
                                                                saver.restore(sess, ckpt.model_checkpoint_path)
                                                                #print('DEBUG')
                                                                #print(saved_args)
                                                                #print(vocab)
								rnnprob *= modeleng.getProbability(sess, eng_chars, eng_vocab, args.sentence, args.n, args.prime,args.sample)[1]
                                                                #print modeleng.getProbability(sess, eng_chars, eng_vocab, args.sentence, args.n, args.prime,args.sample)

					else:
				             print 'Hindi segment', hindisegment
					     
					     englishsegment = tokens[i] + ' '
					     #print 'Inside Hi', tokens[i],englishsegment
					     #changedsen += ' ' + hindisegment
					     if args.matric_lang == 'English':
						args.save_dir = args.hindi_save_dir
						args.data = hindisegment
						translated = translate(args)
						args.data = translated[0]
						if len(args.data) == 0:
							continue
						transliterated = transliterate(args)
						args.sentence = transliterated[0]
						args.n = len(transliterated[0])
						print args.sentence
						changedsen += ' ' + args.sentence
						#if args.n == 0:
						#	continue
						with tf.Session() as sess:
        						tf.global_variables_initializer().run()
        						saver = tf.train.Saver(tf.global_variables())
        						ckpt = tf.train.get_checkpoint_state(args.save_dir)
        						if ckpt and ckpt.model_checkpoint_path:
            							saver.restore(sess, ckpt.model_checkpoint_path)
            							#print('DEBUG')
            							#print(saved_args)
            							#print(vocab)
								rnnprob *= modelhindi.getProbability(sess, hindi_chars, hindi_vocab, args.sentence, args.n, args.prime,args.sample)[1]
            							#print modelhindi.getProbability(sess, hindi_chars, hindi_vocab, args.sentence, args.n, args.prime,args.sample)

						#print calc.calculateprob(args)
						#print transliterated

				else:
					if segments[i] == 'HI':
						hindisegment += tokens[i] + ' '
						englishsegment = ''
					elif segments[i] == 'EN' or segments[i] == 'HASH':
						englishsegment += tokens[i] + ' '
						hindisegment = ''
					#else:
					#	if i!=0 and segments[i-1] == 'EN':
			#print "segment", englishsegment, hindisegment				
						
			if len(englishsegment) > 0 :
				changedsen += englishsegment
			if len(hindisegment) > 0:
				changedsen += hindisegment
			print line, segments, changedsen, rnnprob, prob
		        #break	

if __name__=='__main__':
	main()
