import collections, nltk

filename = "train100k.txt"
string_tuples = open(filename, "r").read().split("\n")
tuples = [x.split("\t") for x in string_tuples if x]

tokens = []
for i in range(len(tuples)):
	tokens.extend(nltk.word_tokenize(tuples[i][1]))
	tokens.extend(nltk.word_tokenize(tuples[i][2]))


def compute_unigram(tokens):
	model = collections.defaultdict(lambda: 0.01)
	for f in tokens:
		try:
			model[f] += 1
		except KeyError:
			model[f] = 1
			continue
	N = float(sum(model.values()))
	for word in model:
		model[word] = model[word]/N
	return model




unigram_prob = compute_unigram(tokens)
print(unigram_prob.keys())