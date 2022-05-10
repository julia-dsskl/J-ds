""" Game called guess the number in less than 20 attempts.
The computer itself thinks and guesses the numbers"""

import numpy as np

def random_predict(number: int = np.random.randint(1, 101)) -> int: 
    """Рандомно угадываем число

    Args:
        number (int, optional): Загаданное число. Defaults to 1.

    Returns:
        int: Число попыток
    """
    
    count = 0
    max_num = 100
    min_num = 0
    predict_number = np.random.randint(1, 101)
    
    while True: 
        count = count + 1
        
        if predict_number > number:
            max_num = predict_number - 1
            predict_number = (min_num + max_num) // 2
            
            
        elif predict_number < number:
            min_num = predict_number +  1
            predict_number = (min_num + max_num) // 2
            
        else:
            break #to end the game and cycle
        
        print(predict_number, number)
        
        return count 
    
def score_game(random_predict) -> int:
    """За какое количство попыток в среднем за 100 подходов угадывает наш алгоритм

    Args:
        random_predict ([type]): функция угадывания

    Returns:
        int: среднее количество попыток
    """
    
    count_ls = []
    np.random.seed(1)
    random_array = np.random.randint(1, 101, size =(250))
    print(random_array)
    print(len(random_array))
    
    i = 0
    for number in random_array:
        i = i + 1
        
    count_ls. append(random_predict(number))
    print(count_ls)
    print(i)
    
    score = int(np.mean(count_ls)) #среднее количество попыток
    print(f'Среднее количество попыток алгоритма угадать число:{score} попыток')
    return(score)

score_game(random_predict)
    