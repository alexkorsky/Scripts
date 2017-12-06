def pretty_print_review_and_label(i):
    print(labels[i] + "\t:\t" + reviews[i][:80] + "...")

g = open('reviews.txt','r') # What we know!
reviews = list(map(lambda x:x[:-1],g.readlines()))
g.close()

g = open('labels.txt','r') # What we WANT to know!
labels = list(map(lambda x:x[:-1].upper(),g.readlines()))
g.close()

print("labels.txt \t : \t reviews.txt\n")
pretty_print_review_and_label(2137)
pretty_print_review_and_label(12816)
pretty_print_review_and_label(6267)
pretty_print_review_and_label(21934)
pretty_print_review_and_label(5297)
pretty_print_review_and_label(4998)


from collections import Counter
import numpy as np

# Create three Counter objects to store positive, negative and total counts
positive_counts = Counter()
negative_counts = Counter()
total_counts = Counter()

# Loop over all the words in all the reviews and increment the counts in the appropriate counter objects
for i in range(len(reviews)):
    if(labels[i] == 'POSITIVE'):
        for word in reviews[i].split(" "):
            positive_counts[word] += 1
            total_counts[word] += 1
    else:
        for word in reviews[i].split(" "):
            negative_counts[word] += 1
            total_counts[word] += 1
    
# cool most_comon function returning most_common dictionary        
print(positive_counts.most_common()[0:5])
print(negative_counts.most_common()[0:5])

pos_neg_ratios = Counter()

# Calculate the ratios of positive and negative uses of the most common words
# Consider words to be "common" if they've been used at least 100 times
for term,cnt in list(total_counts.most_common()):
    if(cnt > 100):
        pos_neg_ratio = positive_counts[term] / float(negative_counts[term]+1)
        pos_neg_ratios[term] = pos_neg_ratio
        
print("Pos-to-neg ratio for 'the' = {}".format(pos_neg_ratios["the"]))
print("Pos-to-neg ratio for 'amazing' = {}".format(pos_neg_ratios["amazing"]))
print("Pos-to-neg ratio for 'terrible' = {}".format(pos_neg_ratios["terrible"]))

'''

    Right now, 1 is considered neutral, but the absolute value of the postive-to-negative 
    rations of very postive words is larger than the absolute value of the ratios for 
    the very negative words. So there is no way to directly compare two numbers and see if 
    one word conveys the same magnitude of positive sentiment as another word conveys 
    negative sentiment. So we should center all the values around netural so the absolute 
    value fro neutral of the postive-to-negative ratio for a word would indicate how much 
    sentiment (positive or negative) that word conveys.
    When comparing absolute values it's easier to do that around zero than one.

To fix these issues, we'll convert all of our ratios to new values using logarithms.

Go through all the ratios you calculated and convert them to logarithms. (i.e. use np.log(ratio))

In the end, extremely positive and extremely negative words will have positive-to-negative
 ratios with similar magnitudes but opposite signs.
'''

# Convert ratios to logs
for word,ratio in pos_neg_ratios.most_common():
    if (ratio > 1):
        pos_neg_ratios[word] = np.log(ratio)
    else:
        pos_neg_ratios[word] =  -np.log(1/(ratio + 0.01))
        
'''
These won't give you the exact same results as the simpler 
code we show in this notebook, but the values will be similar.
 In case that second equation looks strange, here's what it's doing: 
 First, it divides one by a very small number, which will produce a larger 
 positive number. Then, it takes the log of that, which produces numbers similar 
to the ones for the postive words. Finally, it negates the values by adding that
 minus sign up front. The results are extremely positive and extremely negative words 
 having positive-to-negative ratios with similar magnitudes but oppositite signs, 
 just like when we use np.log(ratio).
'''

#EXamine the new ratios you've calculated for the same words from before:
print("Pos-to-neg ratio for 'the' = {}".format(pos_neg_ratios["the"]))
print("Pos-to-neg ratio for 'amazing' = {}".format(pos_neg_ratios["amazing"]))
print("Pos-to-neg ratio for 'terrible' = {}".format(pos_neg_ratios["terrible"]))

# TODO: Create set named "vocab" containing all of the words from all of the reviews
vocab = set(total_counts.keys())
vocab_size = len(vocab)

layer_0 = np.zeros((1, vocab_size))
#shate now is (1, 74074)

# Create a dictionary of words in the vocabulary mapped to index positions
# (to be used in layer_0)
word2index = {}
for i,word in enumerate(vocab):
    word2index[word] = i
    
def update_input_layer(review):
    """ Modify the global layer_0 to represent the vector form of review.
    The element at a given index of layer_0 should represent
    how many times the given word occurs in the review.
    Args:
        review(string) - the string of the review
    Returns:
        None
    """
    global layer_0
    # clear out previous state by resetting the layer to be all 0s
    layer_0 *= 0
    
    # TODO: count how many times each word is used in the given review and store the results in layer_0 
    for word in review.split(" "):
        layer_0[0][word2index[word]] += 1
        
update_input_layer(reviews[0])

print(layer_0)

def get_target_for_label(label):
    """Convert a label to `0` or `1`.
    Args:
        label(string) - Either "POSITIVE" or "NEGATIVE".
    Returns:
        `0` or `1`.
    """
    if(label == 'POSITIVE'):
        return 1
    else:
        return 0