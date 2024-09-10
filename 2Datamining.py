# DATA MINING

#Primeiro Passo, limpeza dos dados: 
#Remover acentuações e tornar minusculas/maiuscula


import pandas as pd

file_path = r"C:\Users\dei20\Downloads\compras_maio-2018 (4).csv"

# Ler o arquivo como um arquivo de texto
with open(file_path, encoding='ISO-8859-1') as file:
    lines = file.readlines()

# Criar um DataFrame com uma coluna 'descricao_compra'
df = pd.DataFrame(lines, columns=['descricao_compra'])

# Remover caracteres de nova linha e espaços extras
df['descricao_compra'] = df['descricao_compra'].str.strip()

# Verificar as primeiras linhas do DataFrame
#print(df.head())

#aqui já havia funcionado


#teremos que transformar em arquivo ascii
#mas, existem opções com outras bibliotecas para isso.

#para transformar vamos usar a biblioteca unidecode
#A biblioteca unicodedata é uma maneira eficiente de remover acentos e transformar caracteres especiais em seus equivalentes ASCII.

#roda no terminal do python abaixo>
#pip install unidecode

# após sua instalação, iremos deixar tudo minuscula e sem acentução

from unidecode import unidecode

def convert_to_ascii(text):
    # Converter texto Unicode para ASCII
    return unidecode(text).lower()

# Caminho para o arquivo CSV #######################################################
file_path = r"C:\Users\dei20\Downloads\compras_maio-2018 (4).csv"

# Ler o arquivo como um arquivo de texto
with open(file_path, encoding='ISO-8859-1') as file:
    lines = file.readlines()

# Criar um DataFrame com uma coluna 'descricao_compra'
df = pd.DataFrame(lines, columns=['descricao_compra'])

# Remover caracteres de nova linha e espaços extras
df['descricao_compra'] = df['descricao_compra'].str.strip()

# Verificar as primeiras linhas do DataFrame
#print(df.head())



#######################################################################

# Aplicar a conversão para ASCII e transformação para minúsculas
df['descricao_compra'] = df['descricao_compra'].apply(convert_to_ascii)

# Verificar as primeiras linhas do DataFrame
#print(df.head())

# Exibir as últimas 5 linhas
#print(df.tail())


#remover espaços extras, caso necessário

#df['descricao_compra'] = df['descricao_compra'].str.strip()

# Agora, caso queira verificar os dados completamente ou exportar, segue:

#removi a primeira linha pq ficou duplicada
df = df.iloc[1:].reset_index(drop=True)
#print(df.head())

#output_file_path = r"C:\Users\dei20\Downloads\compras_maio-2018_clean_ascii.csv"

# Exportar o DataFrame para um arquivo CSV
#df.to_csv(output_file_path, index=False, encoding='ascii')
#print(f"\nDataFrame exportado para CSV com ASCII: {output_file_path}")




################################# 
#Seguindo, vamos retirar os caracteres especiais do arquivo

import re

def replace_special_characters(text):
    # Substituir caracteres especiais por espaços
    return re.sub(r'[^\w\s]', ' ', text)  # Mantém letras, números e espaços

# Substituir caracteres especiais por espaços
df['descricao_compra'] = df['descricao_compra'].apply(replace_special_characters)

#consulte
#print(df.head())

######################## TOKENIZAÇÃO

#vamos seguir em frente para a tokenização! A tokenização é o processo de dividir um texto em unidades menores, chamadas tokens. 
# Esses tokens podem ser palavras, frases ou até mesmo caracteres individuais, dependendo do que você deseja analisar.

#Chame no terminal o pip install nltk spacy, após isso siga:

import nltk
from nltk.tokenize import word_tokenize, sent_tokenize

#pode ser necessario fazer o downloado do nltk a partir de :

#nltk.download()           #se a parte de baixo não rodar  instale assim.



# Baixar os pacotes necessários (execute uma vez)
# NLTK é uma biblioteca amplamente utilizada para processamento de linguagem natural em Python e fornece funções para tokenização básica

#nltk.download('punkt')

#teste se funcionou para um token de palavras: 

#text = "Aqui está um exemplo de texto. Vamos tokenizar esse texto em palavras."

#word_tokens = word_tokenize(text)
#print("Tokens de palavras:", word_tokens)

# VOLTANDO AO CODIGO  #############################################################

#####################
#Pode tokenizar em frases ou palavras, fica ao critério e da necessidade, mas aqui seguiremos com frases

# Tokenização por frases
df['sent_tokens'] = df['descricao_compra'].apply(sent_tokenize)

# Verificar as primeiras linhas do DataFrame após tokenização

#print("Primeiras linhas com tokens de frases:")
#print(df[['sent_tokens']].head())

# Tokenização de palavras, SE QUISER
#df['word_tokens'] = df['descricao_compra'].apply(word_tokenize)

# Verificar as primeiras linhas do DataFrame após tokenização
#print("\nPrimeiras linhas do DataFrame após tokenização:")
#print(df.head())

#Ok, até aqui deu tudo certo, vamos seguir:


#iremos remover palavras curtas de até 3 caracteres

# Função para remover palavras curtas

# com base na nossa tokenização
# Função para remover palavras curtas, até 3 caracteres


def remove_short_words(tokens, max_length=3):
    return [word for word in tokens if len(word) > max_length]


def process_sentences(sentences):
    # Tokenizar cada frase em palavras
    all_tokens = [word for sentence in sentences for word in word_tokenize(sentence)]
    # Remover palavras curtas
    return remove_short_words(all_tokens)

df['filtered_tokens'] = df['sent_tokens'].apply(process_sentences)

# Exibir as primeiras linhas do DataFrame após processamento
#print("\nPrimeiras linhas do DataFrame após remoção de palavras curtas:")
#print(df[['filtered_tokens']].head())

# Ok

# Iremos remover stop words em portugues


#stop words são Stop words são palavras comuns que geralmente são filtradas em tarefas de processamento de linguagem natural (NLP)
#  porque são muito frequentes e não carregam muito significado útil para análises específicas. Essas palavras incluem artigos, 
# preposições, conjunções e outras palavras que não contribuem significativamente para o entendimento do conteúdo textual quando usadas
#  sozinhas. ex: de, a , o , uma, um ...

#nem sempre ele remove corretamente, então talvez criar uma lista personalizada seja melhor
#aqui tentaremos seguir com o que temos no pacote


from nltk.corpus import stopwords
nltk.download('stopwords')

# Carregar stop words em português

stop_words = set(stopwords.words('portuguese'))

def remove_stop_words(tokens):
    # Carregar stop words em português
    stop_words = set(stopwords.words('portuguese'))
    # Remover stop words
    return [word for word in tokens if word.lower() not in stop_words]

def process_sentences_for_stop_words(sentences):
    # Tokenizar cada frase em palavras
    all_tokens = [word for sentence in sentences for word in word_tokenize(sentence, language='portuguese')]
    # Remover stop words
    return remove_stop_words(all_tokens)

df['filtered_tokens'] = df['sent_tokens'].apply(process_sentences_for_stop_words)

#print(df[['filtered_tokens']].head())

# STEMIZAÇÃO  ############################################################
#A stemização (ou stemming) é o processo de reduzir palavras às suas raízes ou radicais. 
# Por exemplo, as palavras "correndo", "corredor" e "corre" poderiam ser reduzidas à raiz comum "corr".

from nltk.stem import RSLPStemmer

# Inicializar o stemmer para português
stemmer = RSLPStemmer()

def stem_tokens(tokens):
    # Aplicar stemização a cada token
    return [stemmer.stem(word) for word in tokens]

def process_sentences_for_stemming(sentences):
    # Tokenizar cada frase em palavras
    all_tokens = [word for sentence in sentences for word in word_tokenize(sentence, language='portuguese')]
    # Remover stop words
    tokens_without_stopwords = remove_stop_words(all_tokens)
    # Aplicar stemização
    return stem_tokens(tokens_without_stopwords)

# Aplicar a função de stemização após a remoção de stop words
df['stemmed_tokens'] = df['sent_tokens'].apply(process_sentences_for_stemming)

# Verificar as primeiras linhas do DataFrame após stemização
print(df[['stemmed_tokens']].head())


# OK


# vamos aplicar o LDA (Latent Dirichlet Allocation) para modelagem de tópicos em seu conjunto de dados. 
# O LDA é uma técnica popular de modelagem de tópicos que descobre temas latentes em um conjunto de documentos.

#no terminal instale pip install gensim scikit-learn

import pandas as pd
from gensim import corpora, models
from gensim.models import CoherenceModel
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import word_tokenize

# Supõe-se que df já possui a coluna 'stemmed_tokens' com tokens já processados

# Passo 2: Transformar os Dados
# Recriar uma coluna de texto a partir dos tokens para o CountVectorizer
df['text'] = df['stemmed_tokens'].apply(lambda x: ' '.join(x))

# Usar CountVectorizer para transformar os dados
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['text'])

# Passo 3: Aplicar o Modelo LDA
# Criar o dicionário e corpus para o LDA
texts = df['stemmed_tokens'].tolist()
dictionary = corpora.Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]

# Aplicar LDA
#lda_model = models.LdaModel(corpus, num_topics=5, id2word=dictionary, passes=15)
lda_model = models.LdaModel(corpus, num_topics=10, id2word=dictionary, passes=15)

# Passo 4: Visualizar os Resultados
topics = lda_model.print_topics(num_words=5)
for topic in topics:
    print(topic)


##################  ajuste como for necessario

# mudando manualmente os parâmetros do modelo LDA
num_topics = 20
alpha = 0.1
beta = 0.01
passes = 10

# Treinar o modelo LDA
lda_model = models.LdaModel(
    corpus=corpus,
    id2word=dictionary,
    num_topics=num_topics,
    alpha=alpha,
    eta=beta,
    passes=passes,
    random_state=42  # Para reprodutibilidade
)

# Mostrar os tópicos
#topics = lda_model.print_topics(num_words=10)

#for topic in topics:
    #print(topic)



#################### visualização em nuvem de palavras
#Por fim, geramos uma nuvem de palavras que calcula o peso de cada tema dentro do topico e exibe quais os principais
# tornando possível entender ao que se refere cada topico.



from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Função para gerar e exibir nuvem de palavras

def plot_wordcloud(topic_words, topic_id):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(topic_words)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f'Tópico {topic_id}')  # Adiciona um título com o ID do tópico
    plt.show()

# Gerar e exibir nuvem de palavras para cada tópico
for topic_id in range(20):  # Ajuste o número conforme o número de tópicos
    words = lda_model.show_topic(topic_id, 10)
    topic_words = ' '.join([word for word, _ in words])
    plot_wordcloud(topic_words, topic_id)  # Passe o ID do tópico para a função   


#Acabamos aqui <3 
