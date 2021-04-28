import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.core.lightning import LightningModule
import os
#from datasets import load_metric
from transformers import AdamW, BartTokenizer, BartForConditionalGeneration, get_linear_schedule_with_warmup
import numpy as np
from compute_rouge import calculate_rouge
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
import string
from collections import Counter
import spacy 
import re
from tqdm import tqdm
import en_core_web_sm
nlp = en_core_web_sm.load()
# stop_words = stopwords.words('english') + list(string.punctuation)
# print(stop_words, "stopwords")

def lmap(f, x):
    """list(map(f, x))"""
    return list(map(f, x))

def shift_tokens_right(input_ids, pad_token_id):
	"""Shift input ids one token to the right, and wrap the last non pad token (usually <eos>)."""
	prev_output_tokens = input_ids.clone()
	index_of_eos = (input_ids.ne(pad_token_id).sum(dim=1) - 1).unsqueeze(-1)
	prev_output_tokens[:, 0] = input_ids.gather(1, index_of_eos).squeeze()
	prev_output_tokens[:, 1:] = input_ids[:, :-1]
	return prev_output_tokens

class BARTSummarization(LightningModule):

    def __init__(self, model_name, tokenizer, learning_rate=1e-4, eval_beams=4):
        super().__init__()
        self.model = BartForConditionalGeneration.from_pretrained(model_name)
        self.tokenizer = tokenizer
        self.model.resize_token_embeddings(len(self.tokenizer))
        #### Check this
        self.decoder_start_token_id = None
        self.eval_beams = eval_beams
        self.learning_rate = learning_rate
        self.all_generated = []
        self.all_actual = []

    def forward(self, input_ids, **kwargs):
        return self.model(input_ids, **kwargs)

    def _step(self, batch):
        pad_token_id = self.tokenizer.pad_token_id
        input_ids = batch['input_ids']
        # print(len(input_ids), "at training")
        attention_mask = batch['attention_mask']
        tgt_ids = batch['tgt_ids']
        decoder_input_ids = tgt_ids
        # decoder_input_ids = shift_tokens_right(tgt_ids, pad_token_id)
        outputs = self(input_ids, attention_mask=attention_mask, decoder_input_ids=decoder_input_ids, use_cache=False)
        lm_logits = outputs[0]
        ce_loss_fct = torch.nn.CrossEntropyLoss(ignore_index=pad_token_id)
        loss_tensors = ce_loss_fct(lm_logits.view(-1, lm_logits.shape[-1]), tgt_ids.view(-1))
        return loss_tensors


    def training_step(self, batch, batch_idx):
        pad_token_id = self.tokenizer.pad_token_id
        loss_tensors = self._step(batch)
        self.log('train_loss', loss_tensors, on_step=True, on_epoch=True, prog_bar=True)
        return {"loss": loss_tensors}

    def ids_to_clean_text(self, ids):
        gen_text = self.tokenizer.batch_decode(
            ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        return lmap(str.strip, gen_text)

    def parse_score(self, result):
        return {k: round(v.mid.fmeasure * 100, 4) for k, v in result.items()}

    def generative_step(self, batch):
        generated_ids = self.model.generate(
            batch["input_ids"],
            attention_mask=batch["attention_mask"],
            use_cache=True,
            decoder_start_token_id=self.decoder_start_token_id,
            num_beams=self.eval_beams,
            length_penalty=2.0,
            max_length=256,
            min_length=10,
            no_repeat_ngram_size=3
        )
        generated_text = self.ids_to_clean_text(generated_ids)
        actual_text = self.ids_to_clean_text(batch['tgt_ids'])
        return generated_text, actual_text

    def validation_step(self, batch, batch_idx):
        loss_tensors = self._step(batch)
        self.log('val_loss', loss_tensors, on_epoch=True, prog_bar=True)
        generated_text, actual_text = self.generative_step(batch)
        self.all_generated += generated_text
        self.all_actual += actual_text
        return {"val_loss": loss_tensors}
    
    def test_step(self, batch, batch_idx):
        loss_tensors = self._step(batch)
        return {"test_loss": loss_tensors}
    #### Validation end and test end methods


    def validation_epoch_end(self, outputs):
        
        avg_val_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        print(calculate_rouge(self.all_generated, self.all_actual,rouge_keys=['rouge1','rouge2', 'rougeL', 'rougeLsum'])) 
        self.all_generated = []
        self.all_actual = []
        self.log("avg_val_loss", avg_val_loss)
        #return {"val_loss": avg_val_loss}

    def configure_optimizers(self):
        "Prepare optimizer and schedule (linear warmup and decay)"

        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)]
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.learning_rate)
        self.opt = optimizer
        return [optimizer]


class TorchDataset(torch.utils.data.Dataset):
    def __init__(self, articles, highlights, tokenizer, max_length=512): ####
        self.x = articles
        self.y = highlights
        # self.length_tokens = length_tokens
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __getitem__(self, index):
        x = self.tokenizer.encode_plus(self.x[index].lower(), max_length=self.max_length, truncation=True, return_tensors="pt", padding='max_length')
        x_input_ids = x['input_ids'].view(-1)
        x_attention_mask = x['attention_mask'].view(-1)

        # x_input_ids = torch.cat((torch.tensor([self.length_tokens[index]]), x['input_ids'].view(-1)), dim=0)
        # x_attention_mask = torch.cat((torch.tensor([1]), x['attention_mask'].view(-1)), dim=0)

        y = self.tokenizer.encode(self.y[index].lower(), max_length=self.max_length, return_tensors="pt", truncation=True, padding='max_length')
        return {'input_ids' : x_input_ids, 'attention_mask' : x_attention_mask, 'tgt_ids' : y.view(-1)}

    def __len__(self):
        return len(self.x)

def read_data(file_path, split):
    if split=="val":
        string_tuples = open("{}{}".format(file_path, "dev1.txt"), "r").read().split("\n")
        string_tuples += open("{}{}".format(file_path, "dev2.txt"), "r").read().split("\n")
    else:
        string_tuples = open("{}{}".format(file_path, split+".txt"), "r").read().split("\n")    
    
    relation = []
    phraseA  = []
    phraseB = []

    tuples = [x.split("\t") for x in string_tuples if x]
    relation = ["<{}>".format(i[0]) for i in tuples]
    phraseA = [i[1] for i in tuples]
    phraseB = [i[2] for i in tuples]

    assert len(phraseA) == len(phraseB)
    assert len(phraseA) == len(relation)

    return phraseA[:10], phraseB[:10], relation[:10] ####

# def find_length_tokens(summary, bin_l, tokenizer):
#     tokens = []
#     for s in summary:
#         curr_len = len(nltk.word_tokenize(s))
#         idx = 0
#         while(idx<10):
#             if(bin_l[idx]>curr_len):
#                 break
#             idx += 1
        
#     return tokens
    



# def find_length_control_bins(train_summary):
#     all_lengths = [len(nltk.word_tokenize(s)) for s in train_summary]
#     N = len(all_lengths)
#     bin_l = [0]
#     all_lengths.sort()
#     for i in range(1,10):
#         bin_l.append(all_lengths[int(i*N/10)])
#     return bin_l

# def replace(old_phrase, new_phrase, sentence):
# 	old_phrase = re.escape(old_phrase)
# 	old_phrase = "((?<=[^a-zA-Z0-9])|(?<=^))" + old_phrase + "((?=[^a-zA-Z0-9])|(?=$))"
# 	return re.sub(old_phrase, new_phrase, sentence, flags=re.IGNORECASE)
	

# def anonymize_data(articles, highlights, filename, output_dir):
# 	anon_articles = []
# 	anon_highlights = []
# 	anon_entities = []
# 	articles_file = open(output_dir + filename+"_articles.txt", "a")
# 	highlights_file = open(output_dir + filename+"_highlights.txt", "a")
# 	entities_file = open(output_dir + filename+ "_entities.txt", "a")
# 	allowed_entitylabels = ["PERSON", "GPE", "ORG", "WORK_OF_ART", "GEO", "NORP", "EVENT"]

# 	print("length of the articles is ", len(highlights))
# 	for article, highlight in tqdm(zip(articles, highlights)):
# 		# print(" article :\n")
# 		# print(article)
# 		articles_entity = nlp(article)
# 		highlight_entity = nlp(highlight)
# 		entities = [X.text for X in articles_entity.ents if X.label_ in allowed_entitylabels]
# 		entities.extend([X.text for X in highlight_entity.ents if X.label_ in allowed_entitylabels])
# 		# print("\nCounter: ",Counter(entities),"\n")
# 		entities_freq = Counter(entities).most_common(10)
# 		entities_limited = []
# 		for entity_tuple in entities_freq:
# 			entities_limited.append(entity_tuple[0])

# 		final_entities = set()
# 		anon_dict = dict()
# 		index = 0
# 		for entity in entities_limited:
# 			flag = 0
# 			for final_entity in final_entities:
# 				try:
# 					if (nlp(entity)).similarity(nlp(final_entity))>0.95:
# 						# print(entity, final_entity)
# 						anon_dict[entity] = anon_dict[final_entity]
# 						flag = 1
# 						break
# 				except UserWarning:
# 					continue
			
# 			final_entities.add(entity)
# 			if flag==1:
# 				continue
# 			anon_dict[entity] = f'@entity{index}'
# 			index +=1

# 		for key, value in anon_dict.items():
# 			article = replace(key, value, article)
# 			highlight = replace(key, value, highlight)

# 		entity_line = " ".join(set(re.findall(r"@entity[0-9]+", highlight)))
# 		anon_entities.append(entity_line)
# 		anon_articles.append(entity_line + article)
# 		anon_highlights.append(highlight)

# 		articles_file.write(entity_line + article+"\n")
# 		highlights_file.write(highlight+"\n")
# 		entities_file.write(entity_line+"\n")

# 		# print("\n anonymized article :\n")
# 		# print(article)
# 		# print("\n anonymized highlight :\n")
# 		# print(highlight)
# 		# print("\n entity string :\n")
# 		# print(anon_entities[-1])
# 		# print("\n")
	
# 	articles_file.close()
# 	highlights_file.close()
# 	entities_file.close()

# 	assert len(anon_entities) == len(anon_articles)
# 	assert len(anon_articles) == len(anon_highlights)
# 	exit()
# 	quit()
# 	return anon_articles, anon_highlights, anon_entities

# format_data(train_phraseA, train_phraseB, train_relations, "formatted_train", output_dir)
def format_data(phraseA, phraseB, relations, filename, output_dir):
    formatted_phraseA = []
    formatted_phraseB = []
    formatted_phraseA_file = open(output_dir + filename + "_phraseA.txt", "w")
    formatted_phraseB_file = open(output_dir + filename + "_phraseB.txt", "w")
    for phraseA_instance, phraseB_instance, relation in zip(phraseA, phraseB, relations):
        formatted_phraseA.append(relation + " " + phraseA_instance)
        formatted_phraseB.append(phraseB_instance)

        formatted_phraseA_file.write(relation + " " + phraseA_instance + "\n")
        formatted_phraseB_file.write(phraseB_instance + "\n")

    formatted_phraseA_file.close()
    formatted_phraseB_file.close()

    assert len(formatted_phraseA) == len(formatted_phraseB)
    return formatted_phraseA, formatted_phraseB

def getspecialtokens():
    conceptnet_relations = [
    'AtLocation', 'CapableOf', 'Causes', 'CausesDesire',
    'CreatedBy', 'DefinedAs', 'DesireOf', 'Desires', 'HasA',
    'HasFirstSubevent', 'HasLastSubevent', 'HasPainCharacter',
    'HasPainIntensity', 'HasPrerequisite', 'HasProperty',
    'HasSubevent', 'InheritsFrom', 'InstanceOf', 'IsA',
    'LocatedNear', 'LocationOfAction', 'MadeOf', 'MotivatedByGoal',
    'NotCapableOf', 'NotDesires', 'NotHasA', 'NotHasProperty',
    'NotIsA', 'NotMadeOf', 'PartOf', 'ReceivesAction', 'RelatedTo',
    'SymbolOf', 'UsedFor'
    ]
    specialtokens = ["<{}>".format(relation) for relation in conceptnet_relations]
    return specialtokens

def main():

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    DATA_PATH = "./data/conceptnet/"
    

    train_phraseA, train_phraseB, train_relations = read_data(DATA_PATH, 'train100k')
    val_phraseA, val_phraseB, val_relations = read_data(DATA_PATH, 'val')
    test_phraseA, test_phraseB, test_relations = read_data(DATA_PATH, 'test')

    MODEL_NAME = 'sshleifer/distilbart-cnn-6-6' ####
    BATCH_SIZE = 4
    MAX_EPOCHS = 5
    tokenizer = BartTokenizer.from_pretrained(MODEL_NAME)

    output_dir = "conceptnet_bart/"
    formatted_train_phraseA, formatted_train_phraseB = format_data(train_phraseA, train_phraseB, train_relations, "formatted_train", output_dir)
    formatted_val_phraseA, formatted_val_phraseB = format_data(val_phraseA, val_phraseB, val_relations, "formatted_val", output_dir)
    formatted_test_phraseA, formatted_test_phraseB = format_data(test_phraseA, test_phraseB, test_relations, "formatted_test", output_dir)
    specialtokens = getspecialtokens()
    tokenizer.add_tokens(specialtokens)
    


    train_ds = TorchDataset(formatted_train_phraseA, formatted_train_phraseB, tokenizer)
    val_ds = TorchDataset(formatted_val_phraseA, formatted_val_phraseB, tokenizer)
    test_ds = TorchDataset(formatted_test_phraseA, formatted_test_phraseB, tokenizer)

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE, num_workers=8)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=BATCH_SIZE, num_workers=8)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=BATCH_SIZE, num_workers=8)

    logger = pl.loggers.TensorBoardLogger('tb_logs', name='BARTSummarization')
    summarization_model = BARTSummarization(MODEL_NAME, tokenizer)
    checkpoint_callback = ModelCheckpoint(
        dirpath='checkpoint_dir',
        save_top_k=-1,
        verbose=True,
        monitor='avg_val_loss',
        mode='min'
    )
    trainer = pl.Trainer(max_epochs=MAX_EPOCHS, logger=logger, gpus=0, callbacks=[checkpoint_callback]) #--
    trainer.fit(summarization_model, train_loader, val_loader)

    model = summarization_model.model
    model.eval()
    model.to(device)
    generated_test_summaries = []
    actual_test_summaries = []
    for i, batch in enumerate(test_loader):
        for k in batch:
            batch[k] = batch[k].to(device)
        generated_text, actual_text = summarization_model.generative_step(batch)
        generated_test_summaries += generated_text
        actual_test_summaries += actual_text
    print(calculate_rouge(generated_test_summaries, actual_test_summaries,rouge_keys=['rouge1','rouge2', 'rougeL', 'rougeLsum']))



if __name__ == '__main__':
    main()
