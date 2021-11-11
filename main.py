import inquirer

if __name__ == "__main__":
    questions = [
    inquirer.List('benchmark',
                    message="Which task would you like to benchmark?",
                    choices=['Sentence-level sentiment polarity', 'Word-level sentiment polarity', 'Named entity recognition', 'Dependency parsing', 'Part-of-speech tagging', 'Dialect classification'],
                ),
    ]
    answers = inquirer.prompt(questions)
    print('Performing ', answers['benchmark'], ', this may take a while.')
    
    if answers['benchmark'] == 'Sentence-level sentiment polarity':
        exec(open('./sentence_level_sentiment_polarity.py').read())
    
    if answers['benchmark'] == 'Word-level sentiment polarity':
        print('configured yet')
        pass

    if answers['benchmark'] == 'Named entity recognition':
        print('configured yet')
        pass

    if answers['benchmark'] == 'Dependency parsing':
        print('configured yet')
        pass

    if answers['benchmark'] == 'Part-of-speech tagging':
        print('configured yet')
        pass

    if answers['benchmark'] == 'Dialect classification':
        exec(open('./DialectClassification.py').read())
    

    