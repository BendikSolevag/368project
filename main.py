import inquirer

if __name__ == "__main__":
    questions = [
    inquirer.List('benchmark',
                    message="Which task would you like to benchmark?",
                    choices=['Sentence-level sentiment polarity', 'Word-level sentiment polarity', 'Named entity recognition', 'Dependency parsing', 'Part-of-speech parsing', 'Dialect classification'],
                ),
    ]
    answers = inquirer.prompt(questions)
    print('Performing ', answers['benchmark'], ', this may take a while.')
    if answers['benchmark'] == 'Sentence-level sentiment polarity':
        exec(open('./sentence_level_sentiment_polarity.py').read())
