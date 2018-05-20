# Answering Neural Network (Ann)
## Введение
(бла-бла о машинном понимании в целом и QA и SQuAD)

## Архитектура модели R-net
Сеть R-net получает на вход контекст и вопрос по этому контексту и выводит указатель на начало и конец ответа (ответ является подстрокой контекста). Процесс состоит из нескольких этапов:
* Закодировать контекст и вопрос
* Получить представление вопроса для контекста
* Применить self-matching attention для контекста, чтобы получить его конечное представление
* Предсказать интервал контекста, ялвяющийся ответом на вопрос
