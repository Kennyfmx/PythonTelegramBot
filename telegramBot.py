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
