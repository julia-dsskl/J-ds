def read_data():
    data = open('/Users/jurain/Downloads/war_peace_processed.txt', 'rt', encoding='utf-8').read()
    data = data.split('\n')
    return data


# Функция подсчёта частоты употребления слова в тексте
def freq(target_word):  
    new_dict = dict() 
    for i in data: # СТРОКА С ОШИБКОЙ
        if i not in new_dict:
            new_dict[i] = 1
        else:
            new_dict[i] = new_dict.get(i) + 1
    return new_dict.get(target_word, 0)


# Вызов функций
read_data()
freq('война')