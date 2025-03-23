class MathParser():
    def __init__(self):
        self.operation_mapping = {
            "plus": "+",
            "increased by": "+",
            "minus": "-",
            "decreased by": "-",
            "times": "*",
            "multiplied by": "*",
            "divided by": "/",
            "over": "/",
            "to the power of": "**",
            "to the power": "**",
            "squared": "**2",
            "open brackets": "(",
            "opened brackets": "(",
            "close brackets": ")",
            "closed brackets": ")",
            'point': '.',
            'dot': '.',
            'equals to': '=',
            'equals': '=',
            'comma': ',',
            'with regards to': 'wrt',
            'with regard to': 'wrt',
        }

        self.number_mapping = {
        'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4,
        'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9,
        'ten': 10, 'eleven': 11, 'twelve': 12, 'thirteen': 13, 'fourteen': 14,
        'fifteen': 15, 'sixteen': 16, 'seventeen': 17, 'eighteen': 18, 'nineteen': 19,
        'twenty': 20, 'thirty': 30, 'forty': 40, 'fifty': 50, 'sixty': 60,
        'seventy': 70, 'eighty': 80, 'ninety': 90,
        'hundred': 100, 'thousand': 1000, 'million': 1000000,
        'billion': 1000000000, 'trillion': 1000000000000
        }

    def remove_redundant(self, string):
        string = string.replace("-", " ")
        return string
    
    def convert_to_numbers(self, string):
        # Split sentence into tokens
        tokens = string.split()

        result = 0
        temp_result = 0
        new_tokens = []
        for token in tokens:
            if token not in self.number_mapping:
                # If the token isn't a number, add the previous result to new_tokens and reset the counters
                if temp_result != 0 or result != 0:
                    new_tokens.append(str(result + temp_result))
                    result = 0
                    temp_result = 0
                new_tokens.append(token)
            else:
                value = self.number_mapping[token]
                if value >= 100:
                    temp_result *= value
                else:
                    temp_result += value
                if value >= 1000:
                    result += temp_result
                    temp_result = 0

        # Check if there's any remaining numeric result after processing all tokens
        if temp_result != 0 or result != 0:
            new_tokens.append(str(result + temp_result))

        return ' '.join(new_tokens)
    
    def convert_operations(self, string):
        for term, symbol in self.operation_mapping.items():
            if term in string:
                string = string.replace(term, symbol)
        return string
    
    def format_operators(self, string):
        tokens = string.split()
        print(tokens)
        result = ""
        for i in range(len(tokens)):
            if i == 0:
                if tokens[i] in "(-":
                    result += tokens[i]
                elif tokens[i].isnumeric():
                    if tokens[i+1] in "+-=" or tokens[i+1].isalpha():
                        result += tokens[i] + " "
                    else:
                        result += tokens[i]
                else:
                    result += tokens[i] + " "
            elif i != len(tokens)-1:
                if tokens[i] == "+":
                    result += tokens[i] + " "
                elif tokens[i] == "-":
                    if (tokens[i+1] == "(") or \
                       (tokens[i-1].isnumeric() and tokens[i+1].isnumeric()) or \
                       (tokens[i-1] == ")" and tokens[i+1].isnumeric()) or \
                       (tokens[i-1].isalpha() and len(tokens[i-1]) == 1) or \
                       (tokens[i-1] == "**2" and tokens[i+1].isnumeric()):
                        result += tokens[i] + " "
                    else:
                        result += tokens[i]
                elif tokens[i] == "=":
                    result += tokens[i] + " "
                elif tokens[i].isnumeric() or tokens[i] == "**2":
                    if tokens[i+1] in "+-=" or tokens[i+1].isalpha():
                        result += tokens[i] + " "
                    else:
                        result += tokens[i]
                elif tokens[i] in "*/(." or tokens[i] == "**":
                    result += tokens[i]
                elif tokens[i] == ")":
                    if tokens[i+1] in "*/":
                        result += tokens[i]
                    else:
                        result += tokens[i] + " "
                elif tokens[i].isalpha() and len(tokens[i]) == 1:
                    if tokens[i+1] not in "()*/" and tokens[i+1] != "**" and tokens[i+1] != "**2":
                        result += tokens[i] + " "
                    else:
                        result += tokens[i]
                else:
                    result += tokens[i] + " "
            else:
                result += tokens[i]
        return result.strip()
    
    def parse(self, string):
        print("Original: ", string)
        string = self.remove_redundant(string.lower())
        print("Cleaned: ", string)
        string = self.convert_to_numbers(string)
        print("Converted: ", string)
        string = self.convert_operations(string)
        print("Converted operations: ", string)
        string = self.format_operators(string)
        print("Parsed: ", string)
        print()
        return string

    def __call__(self, string):
        return self.parse(string)
    
if __name__ == "__main__":
    mp = MathParser()
    print(mp("What is one hundred thousand ?"))
