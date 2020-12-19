import pandas as pd
import re
import gensim
from nltk import sent_tokenize
# To clean and convert a whatsapp txt file export to a CSV file

# read file by lines
file_path = "sample.txt"
f = open(file_path, 'r')
data = f.readlines()
f.close()

# sanity stats
print('num lines: %s' %(len(data)))

# parse text and create list of lists structure
# remove first whatsapp info message
dataset = data[1:]
cleaned_data = []
for line in dataset:
	# grab the info and cut it out
	date = line.split(",")[0]
	line2 = line[len(date):]
	time = line2.split("-")[0][2:]
	line3 = line2[len(time):]
	name = line3.split(":")[0][4:]
	line4 = line3[len(name):]
	message = line4[6:-1] # strip newline charactor

	#print(date, time, name, message)
	cleaned_data.append([message, name, date, time ])

  
# Create the DataFrame 
df = pd.DataFrame(cleaned_data, columns = ['Message', 'Name', 'Date', 'Time']) 

# check formatting 
if 0:
	print(df.head())
	print(df.tail())

# Save it!
df.to_excel(r'sample.xlsx', index=False)
twb_data = pd.read_excel("sample.xlsx")
data_dict = twb_data.to_dict()
new_data_dict = []
for j in data_dict['Message'].values():
        if str(j) != 'nan':
            new_data_dict.append(gensim.utils.simple_preprocess(j, deacc=True, min_len=3))
new_data_dict
texts = new_data_dict
bigram = gensim.models.Phrases(new_data_dict)
from gensim.utils import lemmatize
from nltk.corpus import stopwords
stops = set(stopwords.words('english'))
def process_texts(texts):
    texts = [[word for word in line if word not in stops] for line in texts]
    texts = [bigram[line] for line in texts]
    texts = [[word.decode("utf-8").split('/')[0] for word in lemmatize(' '.join(line), allowed_tags=re.compile('(NN)'), min_length=5)] for line in texts]
    return texts
train_texts = process_texts(new_data_dict)
from gensim.models import LdaModel
from gensim.models.wrappers import LdaMallet
from gensim.corpora import Dictionary
dictionary = Dictionary(train_texts)
corpus = [dictionary.doc2bow(text) for text in train_texts]
ldamodel = LdaModel(corpus=corpus, num_topics=10, id2word=dictionary)
ldamodel.show_topics()
import pyLDAvis.gensim
visualisation = pyLDAvis.gensim.prepare(ldamodel, corpus, dictionary)
pyLDAvis.save_html(visualisation, 'LDA_Visualization.html')