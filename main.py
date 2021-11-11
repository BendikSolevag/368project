import inquirer
import tokenClassification
import sentence_level_sentiment_polarity
import DialectClassification

def sequence_classification():
    questions = [
        inquirer.List('benchmark',
            message="Which task would you like to benchmark?",
            choices=['Sentence-level sentiment polarity', 'Dialect classification'],
        ),
    ]
    answers = inquirer.prompt(questions)
    print('Performing ', answers['benchmark'], ', this may take a while.')
    
    if answers['benchmark'] == 'Sentence-level sentiment polarity':
        sentence_level_sentiment_polarity.run()

    if answers['benchmark'] == 'Dialect classification':
        DialectClassification.run()


if __name__ == "__main__":
    questions = [
        inquirer.List('benchmark',
            message="Please choose a benchmarking category",
            choices=['Sequence classification', 'Token classification'],
        ),
    ]
    answers = inquirer.prompt(questions)
    if answers['benchmark'] == 'Sequence classification':
        sequence_classification()

    if answers['benchmark'] == 'Token classification':
        tokenClassification.run()



    