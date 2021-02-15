import nltk
import random
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# TODO update BOT_CONFIG and Move to separate file

BOT_CONFIG = {
    'intents': {
        'hello': {
            'examples': ['Привет', 'Добрый день', 'Хайдук', 'Хелло', 'Здравствуйте', 'Приветик'],
            'responses': ['Привет', 'Добрый день', 'Че кого!?']
        },
        'bye': {
            'examples': ['Пока', 'Досвидания', 'Гудбай', 'Гудбай', 'Прощай', 'Я ухожу'],
            'responses': ['Еще увидимся','Обращайся','Шлепай']
        },
        'name': {
            'examples': ['как тебя зовут?', 'какое у тебя имя?'],
            'responses':['слишком много имен'],
        },
        'weather': {
            'examples': ['Какая погода за окном?'],
            'responses':['Слишком хорошая, чтобы тратить время впустую'],
        }
    },
    'failure_phrases': [
        'Не понимаю',
        'Скажи по-другому',
    ]
}

texts = []
intent_names = []

for intent, intent_data in BOT_CONFIG['intents'].items():
    for example in intent_data['examples']:
        texts.append(example)
        intent_names.append(intent)  
        
vectorizer = TfidfVectorizer(ngram_range=(2, 4), analyzer='char')
X = vectorizer.fit_transform(texts)
clf = LinearSVC()
clf.fit(X, intent_names)

def classify_intent(replica):
    intent = clf.predict(vectorizer.transform([replica]))[0]
    # clf.classes_
    examples = BOT_CONFIG['intents'][intent]['examples']
    for example in examples:
        example = clear_text(example)
        if len(example) > 0:       
            if abs(len(example) - len(replica)) / len(example) < 0.2:
                distance = nltk.edit_distance(replica, example)
                if len(example) and distance / len(example) < 0.3:
                    return intent
                
def get_answer_by_intent(intent):
    if intent in BOT_CONFIG['intents']:
        responses = BOT_CONFIG['intents'][intent]['responses']
        return random.choice(responses)

# Generative model

with open('ru.conversations.txt', encoding='utf-8') as dialogues_file:
    dialogues_text = dialogues_file.read()
dialogues = dialogues_text.split('\n\n')

def clear_text(text):
    text = text.lower()
    text = ''.join([char for char in text if char in 'абвгдеёжзийклмнопрстуфхцчшщъыьэюя -'])
    return text

dataset = [] # [[question, answer], ...]
questions = set()

for dialogue in dialogues:
    replicas = dialogue.split('\n')
    replicas = replicas[:2]
    
    if len(replicas) == 2:
        question, answer = replicas
        question = clear_text(question[2:])
        answer = answer[2:]
        
        #TODO clear question, answer
        
        if len(question) > 0 and question not in questions:
            questions.add(question)
            dataset.append([question, answer])

dataset_by_word = {} # {word: [[question with word, answer], ...], ...}

for question, answer in dataset:
    words = question.split(' ')
    for word in words:
        if word not in dataset_by_word:
            dataset_by_word[word] = []
        dataset_by_word[word].append([question, answer])
        
dataset_by_word_filtered = {}
for word, word_dataset in dataset_by_word.items():
    word_dataset.sort(key=lambda pair: len(pair[0]))
    dataset_by_word_filtered[word] = word_dataset[:1000]
    
def generate_answer(replica):
    replica = clear_text(replica)
    if not replica:
        return
    words = set(replica.split(' '))
    words_dataset = []
    for word in words:
        if word in dataset_by_word_filtered:
            word_dataset = dataset_by_word_filtered[word]
            words_dataset += word_dataset
    results = [] # [[question, answer, distance], ...]        
    for question, answer in dataset:
        if abs(len(question) - len(replica)) / len(question) < 0.2:
            distance = nltk.edit_distance(replica, question)
            if len(question) and distance / len(question) < 0.2:
                results.append([question, answer, distance])
    question, answer, distance = min(results, key=lambda three: three[2])
    return answer

def get_stub():
    failure_phrases = BOT_CONFIG['failure_phrases']
    return random.choice(failure_phrases)

stats = {'intent': 0, 'generative': 0, 'stubs': 0}

def bot(replica):
    # NLU
    intent = classify_intent(replica)
    
    # generate answer
    
    # rules
    if intent:
        answer = get_answer_by_intent(intent)
        if answer:
            stats['intent'] += 1
            return answer
    # bot generative model
    answer = generate_answer(replica)
    if answer:
        stats['generative'] += 1
        return answer
    # stub
    answer = get_stub()
    stats['stubs'] += 1
    return answer
