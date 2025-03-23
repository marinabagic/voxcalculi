def custom_tokenizer(sentence):
    sentence = sentence.replace(" ", " | ")
    for sym in ['+', '-', '*', '/', '(', ')']:
        sentence = sentence.replace(sym, f' | {sym} | ')
    return sentence.split(' | ')

def calculate_wer(reference, hypothesis):

    reference = custom_tokenizer(reference)
    hypothesis = custom_tokenizer(hypothesis)

    d = [[0 for x in range(len(reference) + 1)] for y in range(len(hypothesis) + 1)]

    for i in range(len(hypothesis) + 1):
        for j in range(len(reference) + 1):
            if i == 0:
                d[0][j] = j
            elif j == 0:
                d[i][0] = i

    for i in range(1, len(hypothesis) + 1):
        for j in range(1, len(reference) + 1):
            if hypothesis[i - 1] == reference[j - 1]:
                substitution_cost = 0
            else:
                substitution_cost = 1

            d[i][j] = min(
                d[i - 1][j] + 1,                                
                d[i][j - 1] + 1,                                
                d[i - 1][j - 1] + substitution_cost             
            )

    return d[len(hypothesis)][len(reference)] / len(reference)

reference = "I am going to roll two fair six-sided dice. Given that the first one shows 5, what is the probability that their sum is 7?"
hypothesis = "I am going to roll 2 fair 6 sided dyces given that the first 1 shows 5 what is the probability that their sum is 7"
print(f"WER: {calculate_wer(reference, hypothesis)}")
