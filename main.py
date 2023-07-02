from collections import defaultdict
import math
import shutil
import time

def estimate_output_probs(delta = 1):
    # Read the training data
    train_file = open("twitter_train.txt", "rb")
    train_data = train_file.read().decode("utf-8")
    train_file.close()
    train_data = train_data.split("\n")

    # Count the number of occurrences of each word/tag
    word_tag_counts = {}
    tag_counts = {}

    for line in train_data:
        if line.strip() == "":
            continue
        word, tag = line.strip().split("\t")

        # Lowercase the word
        word = word.lower()

        # Add "@" in front of words with "USER"
        if "USER" in word and not word.startswith("@"):
            word = "@" + word
        

        if (word,tag) not in word_tag_counts:
            word_tag_counts[(word, tag)] = 1
        else:
            word_tag_counts[(word,tag)] += 1
        #word_tag_counts[(word, tag)] += 1

        #tag_counts[tag] += 1

        if tag not in tag_counts:
            tag_counts[tag] = 1
        else:
            tag_counts[tag] += 1
    #print(f"Final tag counts: {tag_counts}")
    
    lst = []
    for k,v in tag_counts.items():
        lst.append(k)
    print(lst)

    most_common_tag = max(tag_counts, key=tag_counts.get)

    # Calculate the output probabilities
    #num_words = len([word for word, tag in word_tag_counts.keys()])

    # Extract keys (word, tag) from the word_tag_counts dictionary
    keys = word_tag_counts.keys()

    # Create an empty list to store words
    word_list = []
    # Iterate over the keys
    for key in keys:
        # Extract the word from the (word, tag) tuple
        word = key[0]
        # Add the word to the word_list
        word_list.append(word)
 
    #Get unique words
    unique_word_list = set(word_list)

    # Calculate the length of unique_word_set
    num_words = len(unique_word_list)


    output_probs = {}
    for word, tag in word_tag_counts.keys():
        output_probs[(word, tag)] = (word_tag_counts[(word, tag)] + delta) / (tag_counts[tag] + delta * (num_words + 1))

    # Save the output probabilities to a file
    output_file = open("naive_output_probs.txt", "w", encoding="utf-8")
    for word, tag in output_probs.keys():
        output_file.write(tag + "\t" + word + "\t" + str(output_probs[(word, tag)]) + "\n")
    output_file.close()

    final = (lst,most_common_tag)
    return final


# Implement the six functions below
def naive_predict(in_output_probs_filename, in_test_filename, out_prediction_filename):
    #TAGS = ['~', '@', 'O', 'V', '^', ',', '$', 'R', 'A', '!', 'P', 'T', 'N', '&', 'D', '#', 'G', 'U', 'L', 'E', 'X', 'Z', 'S', 'M', 'Y']
    TAGS = estimate_output_probs()[0]
    most_common_tag = estimate_output_probs()[1]
    # Read the output probabilities from file
    output_probs = {}
    with open(in_output_probs_filename, "r", encoding="utf-8") as f:
        for line in f:
            tag, word, prob = line.strip().split("\t")
            output_probs[(word, tag)] = float(prob)

    # Read the test data and make predictions
    test_file = open(in_test_filename, "r", encoding="utf-8")
    predictions = []
    for line in test_file:
        words = line.strip().split()
        pred_tags = []
        for word in words:
            # Lowercase the word
            word = word.lower()

            # Add "@" in front of words with "USER"
            if "USER" in word and not word.startswith("@"):
                word = "@" + word

            max_prob = 0.0
            max_tag = None
            for tag in TAGS:
                if (word, tag) in output_probs:
                    if output_probs[(word, tag)] > max_prob:
                        max_prob = output_probs[(word, tag)]
                        max_tag = tag
            if max_tag is None:
                max_tag = most_common_tag  # Default tag
            pred_tags.append(max_tag)
            #print(pred_tags)
        predictions.append(" ".join(pred_tags))
        
    test_file.close()
    print(len(predictions))


    # Write the predictions to file
    print("Writing predictions to file")
    with open(out_prediction_filename, "w", encoding="utf-8") as f:
        f.write("\n".join(predictions))
    print("Written")

def naive_predict2(in_output_probs_filename, in_train_filename, in_test_filename, out_prediction_filename):
    # Estimate output probabilities and obtain the list of tags
    TAGS = estimate_output_probs()[0]
    most_common_tag = estimate_output_probs()[1]
    
    # Read the output probabilities from the file
    output_probs = {}
    with open(in_output_probs_filename, "r", encoding="utf-8") as f:
        for line in f:
            tag, word, prob = line.strip().split("\t")
            output_probs[(word, tag)] = float(prob)

    # Compute the prior probabilities for the tags from the training data
    tag_counts = {}
    total_tags = 0
    word_counts = {}
    total_words = 0
    with open(in_train_filename, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip() == "":
                continue
            word, tag = line.strip().split("\t")
            

            #tag_counts[tag] += 1 #counts the number of times a tag appears
            #word_counts[word] += 1 #counts the number of times a word appears

            if tag not in tag_counts:
                tag_counts[tag] = 1
            else:
                tag_counts[tag] += 1
            
            if word not in word_counts:
                word_counts[word] = 1
            else:
                word_counts[word] += 1

            total_tags += 1
            total_words += 1


    tag_prob = {} #Each tag_prior is P(y=j)

    for tag, count in tag_counts.items():
        prior_probability = count / total_tags
        tag_prob[tag] = prior_probability


    word_prob = {}
    for word, count in word_counts.items():
        prior_probability = count / total_words
        word_prob[word] = prior_probability


    '''# Compute P(x=w) for each word in the test data
    word_prob = defaultdict(float)
    for (word, tag), prob in output_probs.items():
        word_prob[word] += prob * tag_prob[tag]'''
    


    # Read the test data and make predictions
    test_file = open(in_test_filename, "r", encoding="utf-8")
    predictions = []
    for line in test_file:
        #print("line is", line)
        words = line.strip().split()
        #print("words is", words)
        pred_tags = []
        for word in words:
            # Lowercase the word
            word = word.lower()

            # Add "@" in front of words with "USER"
            if "USER" in word and not word.startswith("@"):
                word = "@" + word
            max_prob = 0.0
            max_tag = None
            for tag in TAGS:
                if (word, tag) in output_probs:
                    prob = output_probs[(word, tag)] * tag_prob[tag] #/word_prob[word]  # P(x = w | y = j) * P(y = j)
                    '''print("output prob:", output_probs[(word, tag)])
                    print("y = j:",tag_prob[tag])
                    print("x = w:",word_prob[word])
                    print(f'prob: {prob}')'''
                    if prob > max_prob:
                        max_prob = prob
                        max_tag = tag
            if max_tag is None:
                max_tag = most_common_tag  # Default tag
            pred_tags.append(max_tag)
        predictions.append(" ".join(pred_tags))
    test_file.close()
    print(len(predictions))

    # Write the predictions to the file
    print("Writing predictions to file")
    with open(out_prediction_filename, "w", encoding="utf-8") as f:
        f.write("\n".join(predictions))
    print("Written")

def estimate_output_q4(delta = None):
    # Read the training data
    train_file = open("twitter_train.txt", "r", encoding="utf-8")
    train_data = train_file.read().split("\n")
    train_file.close()

    # Initialize variables
    word_tag_counts = {}
    tag_counts = {}
    transition_counts = {}

    # Count occurrences of each word-tag pair and tag
    for line in train_data:
        if line.strip() == "":
            continue
        word, tag = line.strip().split("\t")
        word = word.lower()
        if "USER" in word and not word.startswith("@"):
            word = "@" + word

        # Count word-tag pairs
        if (word, tag) not in word_tag_counts:
            word_tag_counts[(word, tag)] = 1
        else:
            word_tag_counts[(word, tag)] += 1

        # Count tags
        if tag not in tag_counts:
            tag_counts[tag] = 1
        else:
            tag_counts[tag] += 1

    # Count transition occurrences
    for i in range(len(train_data)-1):
        if train_data[i].strip() == "" or train_data[i+1].strip() == "": ##DO I Need to include the Stop State
            continue
        tag_i = train_data[i].strip().split("\t")[1]
        tag_j = train_data[i+1].strip().split("\t")[1]
        if (tag_i, tag_j) not in transition_counts:
            transition_counts[(tag_i, tag_j)] = 1
        else:
            transition_counts[(tag_i, tag_j)] += 1

    # Compute output probabilities
    # Iterate over the keys
    word_list = []
    keys = word_tag_counts.keys()
    for key in keys:
        # Extract the word from the (word, tag) tuple
        word = key[0]
        # Add the word to the word_list
        word_list.append(word)

    unique_words = set(word_list)
    num_words = len(unique_words)
    output_probs = {}
    for word, tag in word_tag_counts.keys():
        output_probs[(word, tag)] = (word_tag_counts[(word, tag)] + delta) / (tag_counts[tag] + delta*(num_words+1))

    # Save output probabilities to file
    with open("output_probs.txt", "w", encoding="utf-8") as f:
        for (word, tag), prob in output_probs.items():
            f.write(f"{tag}\t{word}\t{prob}\n")

    # Compute transition probabilities
    transition_probs = {}
    for (tag_i, tag_j), count in transition_counts.items():
        transition_probs[(tag_i, tag_j)] = (count + delta) / (tag_counts[tag_i] + delta*(len(tag_counts)+1))

    # Save transition probabilities to file
    with open("trans_probs.txt", "w", encoding="utf-8") as f:
        for (tag_i, tag_j), prob in transition_probs.items():
            f.write(f"{tag_i}\t{tag_j}\t{prob}\n")

    # Return list of all output tags and the most common output tag
    all_tags = list(tag_counts.keys())
    most_common_tag = max(tag_counts, key=tag_counts.get)
    return all_tags, most_common_tag


def viterbi_predict(in_tags_filename, in_trans_probs_filename, in_output_probs_filename, in_test_filename, out_predictions_filename):
    start_time = time.time()  # Record the start time
    estimate_output_q4(delta=0.01)
    # Load input data
    tags = []
    with open(in_tags_filename, "r", encoding="utf-8") as f:
        for line in f:
            tag = line.strip()
            if tag:
                tags.append(tag)

    trans_probs = {}
    with open(in_trans_probs_filename, "r", encoding="utf-8") as f:
        for line in f:
            tag1, tag2, prob = line.strip().split("\t")
            trans_probs[(tag1, tag2)] = float(prob)

    output_probs = {}
    with open(in_output_probs_filename, "r", encoding="utf-8") as f:
        for line in f:
            tag, word, prob = line.strip().split("\t")
            output_probs[(word, tag)] = float(prob)

    # Load test data
    test_data = []
    with open(in_test_filename, "r", encoding="utf-8") as f:
        words = []
        for line in f:
            word = line.strip().lower()
            if "USER" in word and not word.startswith("@"):
                word = "@" + word
            words.append(word)
            if not line.strip():
                test_data.append(words)
                words = []
        

    def viterbi(obs, states, trans_p, emit_p):
        V = [{}]
        path = {}
        # Initialize base cases (t == 0)
        for state in states:
            V[0][state] = trans_p.get(('START', state), 1e-10) * emit_p.get((obs[0], state), 1e-10)
            path[state] = [state]

        # Run Viterbi for t > 0, computing the maximum probability of reaching each state by multiplying the previous state's probability, 
        # the transition probability to the current state, and the emission probability for the observed word.
        for t in range(1, len(obs)):
            V.append({})
            new_path = {}

            for state in states:
                max_prob = float("-inf")
                prev_state = None

                for y in states:
                    prob = V[t-1][y] * trans_p.get((y, state), 1e-10) * emit_p.get((obs[t], state), 1e-10)
                    if prob > max_prob:
                        max_prob = prob
                        prev_state = y

                V[t][state] = max_prob
                new_path[state] = path[prev_state] + [state]

            path = new_path

        #find the state with the highest probability in the last time step
        max_prob = float("-inf")
        final_state = None

        for state in states:
            if V[-1][state] > max_prob:
                max_prob = V[-1][state]
                final_state = state

        #trace back its path to get the most likely sequence of hidden states.
        return path[final_state]

    # Run Viterbi on test data and write predictions to output file
    with open(out_predictions_filename, "w", encoding="utf-8") as f:
        for sentence in test_data:
            sentence.pop() #Remove empty space at the end
            if sentence == " ":
                continue

            predicted_tags = viterbi(sentence, tags, trans_probs, output_probs)
            for word, tag in zip(sentence, predicted_tags):
                
                f.write(f"{tag}\n")
            f.write("\n")

    end_time = time.time()  # Record the end time

    time_spent = end_time - start_time  # Calculate the time spent
    print("Time spent running the Viterbi: {:.2f} seconds".format(time_spent))

            

       
def replace_number_words(word):
    number_words = {
        "zero": "0",
        "one": "1",
        "two": "2",
        "three": "3",
        "four": "4",
        "five": "5",
        "six": "6",
        "seven": "7",
        "eight": "8",
        "nine": "9",
        "ten": "10",
    }

    if word.lower() in number_words:
        return number_words[word.lower()]
    else:
        return word
    
def replace_special_tokens(word):
    if word.startswith('@'):
        return "@USER"
    elif word.startswith('#'):
        return "#"
    elif word.startswith('http'):
        return "http"
    else:
        return word
    
def find_threshold_at_25th_percentile(frequencies):
        # Create a list of word frequencies sorted in ascending order
        sorted_frequencies = sorted(frequencies.values())
        total_words = len(sorted_frequencies)
        
        # Calculate the 25th percentile index
        percentile_index = int(total_words * 0.25)

        # Find the threshold value at the 25th percentile
        threshold = sorted_frequencies[percentile_index]
        print(threshold)
        
        return threshold


#Q5a)
def estimate_output_q5(delta = None):
    # Read the training data
    train_file = open("twitter_train.txt", "r", encoding="utf-8")
    train_data = train_file.read().split("\n")
    train_file.close()

    # Initialize variables
    word_tag_counts = {}
    tag_counts = {}
    transition_counts = {}
    word_frequencies = {}
    

    # Count occurrences of each word-tag pair and tag
    for line in train_data:
        if line.strip() == "":
            continue
        word, tag = line.strip().split("\t")
        #word = replace_number_words(word)
        word = replace_special_tokens(word)
        word = word.lower()
        if "user" in word and not word.startswith("@"):
            word = "@" + word
        # if word.startswith("#"):
        #     tag = "#"
        # Change the tag to $ for digits
        if word.isdigit():
            tag = "$"
        if word.startswith("http"):
            tag = "U"

        # Count word-tag pairs
        if (word, tag) not in word_tag_counts:
            word_tag_counts[(word, tag)] = 1
        else:
            word_tag_counts[(word, tag)] += 1

        # Update the word_frequencies dictionary
        if word not in word_frequencies:
            word_frequencies[word] = 1
        else:
            word_frequencies[word] += 1

        # Count tags
        if tag not in tag_counts:
            tag_counts[tag] = 1
        else:
            tag_counts[tag] += 1

    


    
    threshold = find_threshold_at_25th_percentile(word_frequencies)  # You can adjust this value based on your needs to calculate proportion of low frequency
    
    # Find the total number of words in the dataset
    total_words = sum(word_frequencies.values())
    # Find the number of unique words in the dataset
    unique_words = len(word_frequencies)
    # Calculate the proportion of low-frequency words (e.g., words appearing less than a certain threshold)
    low_frequency_words = [word for word, count in word_frequencies.items() if count <= threshold]
    low_frequency_word_count = sum([count for word, count in word_frequencies.items() if count <= threshold])
    proportion_low_frequency_words = low_frequency_word_count / total_words
    
    print("Proportion of low frequency words: ", proportion_low_frequency_words)

    # Count transition occurrences
    for i in range(len(train_data)-1):
        if train_data[i].strip() == "" or train_data[i+1].strip() == "": ##DO I Need to include the Stop State
            continue
        tag_i = train_data[i].strip().split("\t")[1]
        tag_j = train_data[i+1].strip().split("\t")[1]
        if (tag_i, tag_j) not in transition_counts:
            transition_counts[(tag_i, tag_j)] = 1
        else:
            transition_counts[(tag_i, tag_j)] += 1

    
    total_words = sum(transition_counts.values())

    # Find the number of unique words in the dataset
    unique_words = len(transition_counts)

    threshold = find_threshold_at_25th_percentile(transition_counts)
    # Calculate the proportion of low-frequency words (e.g., words appearing less than a certain threshold)
    low_frequency_words = [word for word, count in transition_counts.items() if count <= threshold]
    low_frequency_transition_count = sum([count for word, count in transition_counts.items() if count <= threshold])
    proportion_low_frequency_transition = low_frequency_transition_count / total_words
    print("Proportion of low frequency transition: ", proportion_low_frequency_transition )
    

    # Compute output probabilities
    # Iterate over the keys
    word_list = []
    keys = word_tag_counts.keys()
    for key in keys:
        # Extract the word from the (word, tag) tuple
        word = key[0]
        # Add the word to the word_list
        word_list.append(word)
    
    unique_words = set(word_list)
    num_words = len(unique_words)

    count_of_counts = {}  # Initialize the count of counts
    for count in word_tag_counts.values():
        if count not in count_of_counts:
            count_of_counts[count] = 1
        else:
            count_of_counts[count] += 1

    gt_adjusted_counts = {}
    for count in count_of_counts.keys():
        next_count = count + 1
        if next_count in count_of_counts:
            gt_adjusted_counts[count] = (count + 1) * count_of_counts[next_count] / count_of_counts[count]
        else:
            # Handle the missing key by setting a default value
            gt_adjusted_counts[count] = count + 1


   

    output_probs = {}
    for word, tag in word_tag_counts.keys():
        adjusted_count = gt_adjusted_counts[word_tag_counts[(word, tag)]]
        output_probs[(word, tag)] = adjusted_count / tag_counts[tag]

        

    # Extract suffixes and count word-tag pairs with suffixes
    suffix_tag_counts = {}
    for word, tag in word_tag_counts.keys():
        suffix = word[-2:]
        if (suffix, tag) not in suffix_tag_counts:
            suffix_tag_counts[(suffix, tag)] = 1
        else:
            suffix_tag_counts[(suffix, tag)] += 1

    threshold = find_threshold_at_25th_percentile(suffix_tag_counts)
    low_frequency_suffix = [word for word, count in suffix_tag_counts.items() if count <= threshold]
    low_frequency_suffix_count = sum([count for word, count in suffix_tag_counts.items() if count <= threshold])
    proportion_low_frequency_suffix = low_frequency_suffix_count / total_words
    print("Proportion of low frequency suffix: ", proportion_low_frequency_suffix)


    # Compute output probabilities for unseen words based on suffixes
    suffix_output_probs = {}
    for suffix, tag in suffix_tag_counts.keys():
        suffix_output_probs[(suffix, tag)] = (suffix_tag_counts[(suffix, tag)] + delta) / (tag_counts[tag] + delta*(len(unique_words)+1))
    
     # Save suffix output probabilities to file
    with open("suffix_output_probs.txt", "w", encoding="utf-8") as f:
        for (suffix, tag), prob in suffix_output_probs.items():
            f.write(f"{tag}\t{suffix}\t{prob}\n")

    

    # Save output probabilities to file
    with open("output_probs2.txt", "w", encoding="utf-8") as f:
        for (word, tag), prob in output_probs.items():
            f.write(f"{tag}\t{word}\t{prob}\n")

    # Compute transition probabilities
    transition_probs = {}
    for (tag_i, tag_j), count in transition_counts.items():
        transition_probs[(tag_i, tag_j)] = (count + delta) / (tag_counts[tag_i] + delta*(len(tag_counts)+1))

    # Save transition probabilities to file
    with open("trans_probs2.txt", "w", encoding="utf-8") as f:
        for (tag_i, tag_j), prob in transition_probs.items():
            f.write(f"{tag_i}\t{tag_j}\t{prob}\n")

    # Return list of all output tags and the most common output tag
    all_tags = list(tag_counts.keys())
    most_common_tag = max(tag_counts, key=tag_counts.get)
    return all_tags, most_common_tag, suffix_output_probs

'''
Improvements:
Exploiting linguistic pattern: When a word starts with a # or a URL or @USER. We will cluster/group them in my replace_special_tokens
Handling Unseen Words: Improved this by using suffixes

We also introcued using the Good-Turing method, which adjusts the counts based on the frequencies, which is better when dealing with sparse data.
Hence we calculated the frequency of each transitition, word-tag pair and suffix-tag pair and compare it to a threshold.

To calculate this threshold, we have decided to use the 25th percentile of tag counts. 
The counts of each tag is not normally distributed, thus we have chosen against using mean or median counts as an indicator of the threshold.

Proportion of low frequency words:  0.16929790784238224
Proportion of low frequency transition:  0.014570816932532284
Proportion of low frequency suffix:  0.058559491747807474

Using Good-Turing smoothing could be a better choice for word-tag, as the proportion of low-frequency words is relatively high. 
Good-Turing smoothing is specifically designed to handle low-frequency events better than Laplace smoothing, which was the only method used previously.

The Good-Turing method adjusts the probability estimates for low-frequency events by estimating the probability mass 
for unseen events using the counts of events that occurred only once. This is done by using the counts of counts, 
and estimating the probability for unseen events based on the frequency of events that occurred only once/rare/unseen.
This is especially useful when working with word-tag pairs, as natural language often contains many low-frequency words.

On the other hand, Laplace smoothing adds a constant value (delta) to all counts, 
which can help avoid zero probabilities in the estimation. 
It is a simpler technique and might be more suitable for situations where there are not 
many low-frequency events or when the data is relatively uniform.

Hence, we used Good-Turing for output probabilities of word-tag while Laplace for 
output probabilities of unseen words based on suffixes and transition probabilities based on their proportion of low frequency tags.


In summary, we introduced 
1) Special token handling: It replaces user mentions with "@USER", hashtags with "#", and URLs with "http". to help the model better generalize and reduces the sparsity of the feature space.
2) Tag modification for digits and URLs to better recognize and tag numerical values and URLs consistently, which might improve overall tagging accuracy.
3) Suffix-based output probabilities for unseen words: This approach takes advantage of morphological similarities between words and 
can help the model to better estimate probabilities for unseen words or words with rare suffixes.
4) Good-Turing adjusted counts: based on their proportion of low frequency words to assign more accurate probabilities to unseen or rare events
'''


def viterbi_predict2(in_tags_filename, in_trans_probs_filename, in_output_probs_filename, in_test_filename, out_predictions_filename):
    start_time = time.time()  # Record the start time
    all_tags, most_common_tag, suffix_output_probs = estimate_output_q5(delta=1)
    # Load input data
    tags = []
    with open(in_tags_filename, "r", encoding="utf-8") as f:
        for line in f:
            tag = line.strip()
            if tag:
                tags.append(tag)

    trans_probs = {}
    with open(in_trans_probs_filename, "r", encoding="utf-8") as f:
        for line in f:
            tag1, tag2, prob = line.strip().split("\t")
            trans_probs[(tag1, tag2)] = float(prob)

    output_probs = {}
    with open(in_output_probs_filename, "r", encoding="utf-8") as f:
        for line in f:
            tag, word, prob = line.strip().split("\t")
            output_probs[(word, tag)] = float(prob)

    # Load test data
    test_data = []
    with open(in_test_filename, "r", encoding="utf-8") as f:
        words = []
        for line in f:
            word = line.strip()
            #word = replace_number_words(word)
            word = replace_special_tokens(word)
            word = word.lower()
            if "user" in word and not word.startswith("@"):
                word = "@" + word
            words.append(word)
            if not line.strip():
                test_data.append(words)
                words = []
        

    def viterbi(obs, states, trans_p, emit_p, suffix_emit_p):
        V = [{}]
        path = {}
        # Initialize base cases (t == 0)
        for state in states:
            V[0][state] = trans_p.get(('START', state), 1e-10) * emit_p.get((obs[0], state), 1e-10)
            path[state] = [state]

        # Run Viterbi for t > 0
        for t in range(1, len(obs)):
            V.append({})
            new_path = {}

            for state in states:
                max_prob = float("-inf")
                prev_state = None

                for y in states:
                    prob = V[t-1][y] * trans_p.get((y, state), 1e-10) * emit_p.get((obs[t], state), 1e-10)
                    if prob > max_prob:
                        max_prob = prob
                        prev_state = y

                V[t][state] = max_prob
                new_path[state] = path[prev_state] + [state]

            path = new_path

        #find the state with the highest probability in the last time step
        max_prob = float("-inf")
        final_state = None

        for state in states:
            if V[-1][state] > max_prob:
                max_prob = V[-1][state]
                final_state = state

        return path[final_state]

    # Run Viterbi on test data and write predictions to output file
    with open(out_predictions_filename, "w", encoding="utf-8") as f:
        for sentence in test_data:
            sentence.pop() #Remove empty space at the end
            if sentence == " ":
                continue

            predicted_tags = viterbi(sentence, tags, trans_probs, output_probs,suffix_output_probs)
            
            for word, tag in zip(sentence, predicted_tags):
                #f.write(f"{word}\t{tag}\n")
                f.write(f"{tag}\n")
            f.write("\n")

    end_time = time.time()  # Record the end time

    time_spent = end_time - start_time  # Calculate the time spent
    print("Time spent running the Viterbi2: {:.2f} seconds".format(time_spent))



def evaluate(in_prediction_filename, in_answer_filename):
    """Do not change this method"""
    with open(in_prediction_filename) as fin:
        predicted_tags = [l.strip() for l in fin.readlines() if len(l.strip()) != 0]
        print("predicted_tags", len(predicted_tags))

    with open(in_answer_filename) as fin:
        ground_truth_tags = [l.strip() for l in fin.readlines() if len(l.strip()) != 0]
        print("ground_truth_tags", len(ground_truth_tags))

    assert len(predicted_tags) == len(ground_truth_tags)
    correct = 0
    for pred, truth in zip(predicted_tags, ground_truth_tags):
        if pred == truth: correct += 1
    return correct, len(predicted_tags), correct/len(predicted_tags)



def run():
    '''
    You should not have to change the code in this method. We will use it to execute and evaluate your code.
    You can of course comment out the parts that are not relevant to the task that you are working on, but make sure to
    uncomment them later.
    This sequence of code corresponds to the sequence of questions in your project handout.
    '''

    ddir = r'C:\Users\tayzh\Desktop\BT3102\project-files\projectfiles' #your working dir
    
    in_train_filename = f'{ddir}/twitter_train.txt'

    naive_output_probs_filename = f'{ddir}/naive_output_probs.txt'

    in_test_filename = f'{ddir}/twitter_dev_no_tag.txt'
    in_ans_filename  = f'{ddir}/twitter_dev_ans.txt'
    naive_prediction_filename = f'{ddir}/naive_predictions.txt'
    naive_predict(naive_output_probs_filename, in_test_filename, naive_prediction_filename)
    correct, total, acc = evaluate(naive_prediction_filename, in_ans_filename)
    print(f'Naive prediction accuracy:     {correct}/{total} = {acc}')

    naive_prediction_filename2 = f'{ddir}/naive_predictions2.txt'
    naive_predict2(naive_output_probs_filename, in_train_filename, in_test_filename, naive_prediction_filename2)
    correct, total, acc = evaluate(naive_prediction_filename2, in_ans_filename)
    print(f'Naive prediction2 accuracy:    {correct}/{total} = {acc}')

    trans_probs_filename =  f'{ddir}/trans_probs.txt'
    output_probs_filename = f'{ddir}/output_probs.txt'

    in_tags_filename = f'{ddir}/twitter_tags.txt'
    viterbi_predictions_filename = f'{ddir}/viterbi_predictions.txt'
    viterbi_predict(in_tags_filename, trans_probs_filename, output_probs_filename, in_test_filename,
                    viterbi_predictions_filename)
    correct, total, acc = evaluate(viterbi_predictions_filename, in_ans_filename)
    print(f'Viterbi prediction accuracy:   {correct}/{total} = {acc}')

    trans_probs_filename2 =  f'{ddir}/trans_probs2.txt'
    output_probs_filename2 = f'{ddir}/output_probs2.txt'

    viterbi_predictions_filename2 = f'{ddir}/viterbi_predictions2.txt'
    viterbi_predict2(in_tags_filename, trans_probs_filename2, output_probs_filename2, in_test_filename,
                     viterbi_predictions_filename2)
    correct, total, acc = evaluate(viterbi_predictions_filename2, in_ans_filename)
    print(f'Viterbi2 prediction accuracy:  {correct}/{total} = {acc}')
    


if __name__ == '__main__':
    run()
