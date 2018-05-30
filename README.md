# Answering Neural Network (Ann)
## Введение
Исследователи из Стэнфордского университета летом 2016 года представили [специальную базу](https://rajpurkar.github.io/SQuAD-explorer/), состоящую из коротких текстовых фрагментов, вопросов по этим фрагментам и ответов на эти вопросы. Всего в базе, получившей название SQuAD, содержится более 100 тысяч вопросов. По замыслу, с помощью этой базы разработчики систем обработки естественного языка могут тренироваться на задаче поиска ответа по тексту.
Сами авторы SQuAD написали достаточно простой алгоритм, который дает верный ответ в 40 процентах случаев. Для человека этот показатель составляет 82,304 процента. Все желающие могли предложить свою систему, чтобы авторы ее протестировали и опубликовали результат на своем сайте.

С тех пор было предложено 77 систем. Уже в августе 2016 появилось решение с 60 процентами верных ответов, к сентябрю была достигнута отметка в 70 процентов, со временем результаты становились все лучше и лучше и в конце 2017-го — начале 2018 года вплотную приблизились к человеческим. В первые дни 2018 года и было зафиксировано почти одновременное пересечение «финишной ленточки»: 82,65 процента (Microsoft) и 82,44 процента (Alibaba). На данный момент лучший результат у Google Brain & Carnegie Mellon University (83.877 процентов).

## Архитектура модели R-net
Сеть R-net получает на вход контекст и вопрос по этому контексту и выводит указатель на начало и конец ответа (ответ является подстрокой контекста). Процесс состоит из нескольких этапов:
* Закодировать контекст и вопрос
* Получить представление вопроса для контекста
* Применить self-matching attention для контекста, чтобы получить его конечное представление
* Предсказать интервал контекста, ялвяющийся ответом на вопрос
