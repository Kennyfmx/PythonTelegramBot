import nltk
import random
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# TODO Update Config and Move to separate file

BOT_CONFIG = {
    'intents': {
        'hello': { 
            'examples': ['Привет', 'Приветик', 'Приветствую', 'Доброе утро', 'Добрый день', 'Добрый вечер', 'Здравствуй', 'Здравствуйте', 'Хай', 'Хелло'], 
            'responses': ['Привет', 'Приветик', 'Приветствую', 'Доброе утро', 'Добрый день', 'Добрый вечер', 'Здравствуй', 'Здравствуйте', 'Хай', 'Хелло'] 
        },
        'bye': {
            'examples': ['Пока', 'Досвидания', 'Гудбай', 'Гудбай', 'Прощай', 'Я ухожу'],
            'responses': ['Еще увидимся','Обращайся','Шлепай']
        },
        'questions': {
            'examples': ['Сколько тебе лет?'],
            'responses':['Помоложе тебя, старый'],
        },
        'weather': {
            'examples': ['А Женю знаешь?'],
            'responses':['Да, слышала о нем'],
        }
    },
    'failure_phrases': [
        'Не понимаю',
        'Скажи по-другому',
    ]
}

def clear_text(text):
    text = text.lower()
    text = ''.join([char for char in text if char in 'абвгдеёжзийклмнопрстуфхцчшщъыьэюя -'])
    return text

# Levenshtein distance 

def classify_intent(replica):
    replica = clear_text(replica)
    for intent, intent_data in BOT_CONFIG['intents'].items():
        for example in intent_data['examples']:
            example = clear_text(example)
            
            distance = nltk.edit_distance(replica, example)
            
            if distance / len(example) < 0.3:
                return intent
                
def get_answer_by_intent(intent):
    if intent in BOT_CONFIG['intents']:
        responses = BOT_CONFIG['intents'][intent]['responses']
        return random.choice(responses)
        
def genarate_answer(replica):
    # TODO
    return
    
def get_stub():
    failure_phrases = BOT_CONFIG['failure_phrases']
    return random.choice(failure_phrases)

def bot(replica):
    # NLU
    intent = classify_intent(replica)
    
    # generate answer
    
    # rules
    if intent:
        answer = get_answer_by_intent(intent)
        if answer:
            return answer
    # bot generation model
    answer = genarate_answer(replica)
    if answer:
        return answer
    # stub
    answer = get_stub()
    return answer
    
print(bot('привет'))
