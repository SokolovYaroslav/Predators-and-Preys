# Predators & Preys
## Описание окружения
В игре участвуют две команды агентов: жертвы и охотники. Задачей охотников является поимка как можно большего числа жертв, задачей жертв является не быть поймаными. Игра заканчивается или если все жертвы были пойманы, или если вышло отведенное время.

Игра происходит на ограниченном поле с препятствиями. Все сущности для простоты считаются круглыми и генерируются случайно.

Хищники как правило медленнее жертв, а потому им приходится кооперироваться чтобы кого-то поймать.


## Задача
Ваша задача - обучить RL агента, который будет управлять охотниками и жертвами. Для этого необходимо имплементировать наследники классов `PreyAgent` и `PredatorAgent` (файл `predators_and_preys_env/agent.py`). Интерфейс у классов одинаковый, два разных класса нужны только для того, чтобы агент знал, какой командой он управляет.

Исход матча - это количество пойманных жертв для агента-охотника и количество непойманных для агента-жертвы.

Во время обучения можно менять конфигурацию среды при необходимости, однако финальный агент будет выполняться со стандартными настройками среды.

#### State dict
На вход агент получает словарь, описывающий сущности на поле. Словарь содержит внутри три листа, которые соответствуют ключам `obstacles`, `predators` и `preys`.

Для каждой сужности опредлены ее координаты `x_pos` и `y_pos`, а также радиус `radius`. Кроме того, для охотников и жертв также определена скорость `speed`, а для жертв также определено, являются ли они живыми или нет `is_alive`.

#### Action
Совершаемое агентом действие - это набор углов направлений движения всех существ в команде агента. Для удобства углы приведены в диапазон `[-1., 1.]`.

## Лидерборд и evaluation
Оценка уровня агента производится на основе среднего значения пойманых (охотник) и не пойманых (жертва) жертв. Для справедливости оценки, агент играет равное количество матчей за каждую из команд. Также поскольку агенты в лидерборде могут меняться, оценка будет проводиться на основе `300` последних матчей агента.

Соперники для агента выбираются случайным образом.

## Система оценивания
#### Соревнование
Обязательным условием получения баллов за соревновние является нахождение в лидерборде выше, чем baseline агент. Кроме того, решение обязательно должно быть основано на RL.

Далее баллы выставляются следующим образом:
* Топ-20% участников - 10 баллов
* Топ-40% участников - 9 баллов
* Топ-60% участников - 8 баллов
* Топ-80% участников - 7 баллов
* Все остальные - 6 баллов

Кроме того, по итогам соревнования необходимо написать отчет в свободной форме, в котором необходимо описать использованные алгоритмы, представление состояния, функцию награды, эксперименты и т.д., а также приложить исходный код решения.

#### Дополнительные баллы
Дополнительные баллы выдаются на усмотрение лектора. Их можно получить за:

* За красиво оформленный и содержательный отчет можно получить 1 дополнительный балл.
* За имплементацию сложных алгоритмов или необычных идей в рамках соревнования можно получить еще 1 дополнительный балл.

#### Экзамен
За экзамен можно получить до 4 баллов. На экзамене экзаменуемому предлагается ответить на два билета по темам курса.

#### Итоговая оценка 
Итоговая оценка - это сумма всех оценок, но не больше 10 баллов.