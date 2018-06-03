# Answering Neural Network (Ann)
## Введение
Исследователи из Стэнфордского университета летом 2016 года представили [специальную базу](https://rajpurkar.github.io/SQuAD-explorer/), состоящую из коротких текстовых фрагментов, вопросов по этим фрагментам и ответов на эти вопросы. Всего в базе, получившей название SQuAD, содержится более 100 тысяч вопросов. По замыслу, с помощью этой базы разработчики систем обработки естественного языка могут тренироваться на задаче поиска ответа по тексту.
Сами авторы SQuAD написали достаточно простой алгоритм, который дает верный ответ в 40 процентах случаев. Для человека этот показатель составляет 82,304 процента. Все желающие могли предложить свою систему, чтобы авторы ее протестировали и опубликовали результат на своем сайте.

С тех пор было предложено 77 систем. Уже в августе 2016 появилось решение с 60 процентами верных ответов, к сентябрю была достигнута отметка в 70 процентов, со временем результаты становились все лучше и лучше и в конце 2017-го — начале 2018 года вплотную приблизились к человеческим. В первые дни 2018 года и было зафиксировано почти одновременное пересечение «финишной ленточки»: 82,65 процента (Microsoft) и 82,44 процента (Alibaba). На данный момент лучший результат у Google Brain & Carnegie Mellon University (83.877 процентов).

## Модель R-net
Для решения задачи я выборала модель, являющуюся одним из лидеров (топ-4) среди single-моделей. Эта модель носит название R-net. Она разработана исследователями из Microsoft Research Asia и в мае 2017 года они выпустили [статью](https://www.microsoft.com/en-us/research/wp-content/uploads/2017/05/r-net.pdf), описывающую архитектуру модели и некоторые детали реализации.

### Архитектура модели R-net
Сеть R-net получает на вход контекст и вопрос по этому контексту и выводит указатель на начало и конец ответа (ответ является подстрокой контекста). Процесс состоит из нескольких этапов:
* Закодировать контекст и вопрос
* Получить представление вопроса для контекста
* Применить self-matching attention для контекста, чтобы получить его конечное представление
* Предсказать интервал контекста, ялвяющийся ответом на вопрос

 ### Результаты
 Модели оценваются по двум параметрам: ExactMatch (EM) and F1-score (F1). Для человека эти параметры составляют	82.304% и 91.221% соответственно. 
 
 На данный момент (июнь 2018) результаты, которые показала модель R-net - это EM = 81.391 и F1 = 88.170. Как видно, эти результаты близки к человеческим, но всё же их не превосходят.
 
 *А теперь о результатах, которых достигла я:*

Модель обучается очень долго – в среднем 9000 сек. на одной эпохе на графическом процессоре NVIDIA Tesla K80. F1-score составляет примерно 38.8% на dev-set на 6 эпохах. Но на train-set F1-score отличается незначительно . Это означает, что модель не переобучается и, судя по всему, может обучаться достаточно долго (исследователи достигли своих лучших результатов на 31-ой эпохе).

## Инструкции по запуску
* Для начала, установите все необходимые пакеты, описанные в [requirements.txt]()
* Далее необходимо клонировать репозиторий 
```
git clone https://github.com/maduardar/squad
cd squad/
```
* Затем выполните загрузку и предобработку данных:
``` 
sudo python3 prepare.py
``` 
Если у Вас есть уже скачанный 'glove.840B.300d.txt', положите его в папку Data –– это ускорит процеесс
* Для обучения: 
```
sudo python3 train.py
```
* Для тестирование на dev-set и подсчёта F1-score:
```
sudo python3 test.py
```
* Для запуска скрипта, который по входному контексту и вопросу выдаёт ответ:
```
sudo python3 demo.py
```
## Благодарности
Хочу выразить огромную благодарность [Д. А.](https://github.com/dasimagin)[ Симагину](https://stackoverflow.com/), нашему [ментору](https://github.com/maduardar/G2P/blob/master/plm.jpg) и наставнику, за то, что помогал и поддерживал в течение всего проекте.
