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
