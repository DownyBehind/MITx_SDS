from string import punctuation, digits
import numpy as np
import random



#==============================================================================
#===  PART I  =================================================================
#==============================================================================



def get_order(n_samples):
    try:
        with open(str(n_samples) + '.txt') as fp:
            line = fp.readline()
            return list(map(int, line.split(',')))
    except FileNotFoundError:
        random.seed(1)
        indices = list(range(n_samples))
        random.shuffle(indices)
        return indices



def hinge_loss_single(feature_vector, label, theta, theta_0):
    """
    Finds the hinge loss on a single data point given specific classification
    parameters.

    Args:
        `feature_vector` - numpy array describing the given data point.
        `label` - float, the correct classification of the data
            point.
        `theta` - numpy array describing the linear classifier.
        `theta_0` - float representing the offset parameter.
    Returns:
        the hinge loss, as a float, associated with the given data point and
        parameters.
    """
    # Your code here
    agreement = label * (np.dot(feature_vector, theta) + theta_0)
    # claculation of agreement
    hinge_loss = max(0, 1 - agreement)
    return hinge_loss
    raise NotImplementedError



def hinge_loss_full(feature_matrix, labels, theta, theta_0):
    """
    Finds the hinge loss for given classification parameters averaged over a
    given dataset

    Args:
        `feature_matrix` - numpy matrix describing the given data. Each row
            represents a single data point.
        `labels` - numpy array where the kth element of the array is the
            correct classification of the kth row of the feature matrix.
        `theta` - numpy array describing the linear classifier.
        `theta_0` - real valued number representing the offset parameter.
    Returns:
        the hinge loss, as a float, associated with the given dataset and
        parameters.  This number should be the average hinge loss across all of
    """
    agreement = labels * (feature_matrix @ theta + theta_0)
    # @ oprator is a function of matrix multiplication
    """
    import numpy as np
    
    # input
    A = np.array([[1, 2], [3, 4]])
    B = np.array([[5, 6], [7, 8]])
    
    product = A @ B 
    # matrix of multiplication 
    
    print(product)

    # output
    [[19 22]
    [43 50]]
    """
    hinge_loss = np.maximum(0, 1 - agreement)
    # choose maximum values of each elements

    return np.mean(hinge_loss)
    # average values of all elements

    # Your code here
    raise NotImplementedError


def perceptron_single_step_update(
        feature_vector,
        label,
        current_theta,
        current_theta_0):
    """
    Updates the classification parameters `theta` and `theta_0` via a single
    step of the perceptron algorithm.  Returns new parameters rather than
    modifying in-place.

    Args:
        feature_vector - A numpy array describing a single data point.
        label - The correct classification of the feature vector.
        current_theta - The current theta being used by the perceptron
            algorithm before this update.
        current_theta_0 - The current theta_0 being used by the perceptron
            algorithm before this update.
    Returns a tuple containing two values:
        the updated feature-coefficient parameter `theta` as a numpy array
        the updated offset parameter `theta_0` as a floating point number
    """
    epsilon = 10**(-8)
    if label * (feature_vector @ current_theta + current_theta_0) < epsilon:
        # Actually epsilon can be replaced with zero 
        # But why i choose the epsilon is not zero, sometimes the result of the conditions 
        # can be float and between 0 and 1. In this case computer can decide the number is zero.
        # for exceptping this confusion, i use the epsilon. 

        # if the result of the condition is below the epsilon, it can be asssumed that 
        # this classifier line is incorrect, so it will be modified
        current_theta += feature_vector * label
        current_theta_0 += label
        # above two functions are the calculations for modify the classifier line
    return (current_theta, current_theta_0)
    
    # Your code here
    raise NotImplementedError



def perceptron(feature_matrix, labels, T):
    """
    Runs the full perceptron algorithm on a given set of data. Runs T
    iterations through the data set: we do not stop early.

    NOTE: Please use the previously implemented functions when applicable.
    Do not copy paste code from previous parts.

    Args:
        `feature_matrix` - numpy matrix describing the given data. Each row
            represents a single data point.
        `labels` - numpy array where the kth element of the array is the
            correct classification of the kth row of the feature matrix.
        `T` - integer indicating how many times the perceptron algorithm
            should iterate through the feature matrix.

    Returns a tuple containing two values:
        the feature-coefficient parameter `theta` as a numpy array
            (found after T iterations through the feature matrix)
        the offset parameter `theta_0` as a floating point number
            (found also after T iterations through the feature matrix).
    """
    # Your code here
    n_data = feature_matrix.shape[0]
    # the number of datas
    n_para = feature_matrix.shape[1]
    # the number of paras
    theta = np.zeros(n_para)
    theta_0 = 0
    # print(feature_matrix.shape)
    # print("feature_matrix")
    # print(feature_matrix)
    # print("n_data")
    # print(n_data)
    # print("n_para")
    # print(n_para)
    for t in range(T):
        for i in get_order(n_data):
            theta, theta_0 = perceptron_single_step_update(feature_matrix[i],
                                                           labels[i],
                                                           theta,
                                                           theta_0)
    return (theta, theta_0)
    # Your code here
    raise NotImplementedError



def average_perceptron(feature_matrix, labels, T):
    """
    Runs the average perceptron algorithm on a given dataset.  Runs `T`
    iterations through the dataset (we do not stop early) and therefore
    averages over `T` many parameter values.

    NOTE: Please use the previously implemented functions when applicable.
    Do not copy paste code from previous parts.

    NOTE: It is more difficult to keep a running average than to sum and
    divide.

    Args:
        `feature_matrix` -  A numpy matrix describing the given data. Each row
            represents a single data point.
        `labels` - A numpy array where the kth element of the array is the
            correct classification of the kth row of the feature matrix.
        `T` - An integer indicating how many times the perceptron algorithm
            should iterate through the feature matrix.

    Returns a tuple containing two values:
        the average feature-coefficient parameter `theta` as a numpy array
            (averaged over T iterations through the feature matrix)
        the average offset parameter `theta_0` as a floating point number
            (averaged also over T iterations through the feature matrix).
    """
    n_data = feature_matrix.shape[0]
    n_para = feature_matrix.shape[1]
    theta = np.zeros(n_para)
    theta_0 = 0
    theta_mean = np.zeros(n_para)
    theta_0_mean = 0
    for t in range(T):
        for i in get_order(n_data):
            theta, theta_0 = perceptron_single_step_update(feature_matrix[i],
                                                           labels[i],
                                                           theta,
                                                           theta_0)
            theta_mean += theta/(T*n_data)
            theta_0_mean += theta_0/(T*n_data)
    return (theta_mean, theta_0_mean)

    raise NotImplementedError


def pegasos_single_step_update(
        feature_vector,
        label,
        L,
        eta,
        current_theta,
        current_theta_0):
    """
    Updates the classification parameters `theta` and `theta_0` via a single
    step of the Pegasos algorithm.  Returns new parameters rather than
    modifying in-place.

    Args:
        `feature_vector` - A numpy array describing a single data point.
        `label` - The correct classification of the feature vector.
        `L` - The lamba value being used to update the parameters.
        `eta` - Learning rate to update parameters.
        `theta` - The old theta being used by the Pegasos
            algorithm before this update.
        `theta_0` - The old theta_0 being used by the
            Pegasos algorithm before this update.
    Returns:
        a tuple where the first element is a numpy array with the value of
        theta after the old update has completed and the second element is a
        real valued number with the value of theta_0 after the old updated has
        completed.
    """
    if label * (feature_vector @ current_theta + current_theta_0) <= 1:
        current_theta =  (1 - eta*L) * current_theta + eta * feature_vector * label
        current_theta_0 = current_theta_0 + eta * label
    else:
        current_theta = (1 - eta*L) * current_theta
    return (current_theta, current_theta_0)

    raise NotImplementedError



def pegasos(feature_matrix, labels, T, L):
    """
    Runs the Pegasos algorithm on a given set of data. Runs T iterations
    through the data set, there is no need to worry about stopping early.  For
    each update, set learning rate = 1/sqrt(t), where t is a counter for the
    number of updates performed so far (between 1 and nT inclusive).

    NOTE: Please use the previously implemented functions when applicable.  Do
    not copy paste code from previous parts.

    Args:
        `feature_matrix` - A numpy matrix describing the given data. Each row
            represents a single data point.
        `labels` - A numpy array where the kth element of the array is the
            correct classification of the kth row of the feature matrix.
        `T` - An integer indicating how many times the algorithm
            should iterate through the feature matrix.
        `L` - The lamba value being used to update the Pegasos
            algorithm parameters.

    Returns:
        a tuple where the first element is a numpy array with the value of the
        theta, the linear classification parameter, found after T iterations
        through the feature matrix and the second element is a real number with
        the value of the theta_0, the offset classification parameter, found
        after T iterations through the feature matrix.
    """
    n_data = feature_matrix.shape[0]
    n_para = feature_matrix.shape[1]
    theta = np.zeros(n_para)
    theta_0 = 0
    count = 1
    for t in range(T):
        for i in get_order(n_data):
            eta = 1/(count**0.5)
            theta, theta_0 = pegasos_single_step_update(feature_matrix[i],
                                                        labels[i],
                                                        L,
                                                        eta,
                                                        theta,
                                                        theta_0)
            count += 1
    return (theta, theta_0)

    raise NotImplementedError



#==============================================================================
#===  PART II  ================================================================
#==============================================================================



##  #pragma: coderesponse template
##  def decision_function(feature_vector, theta, theta_0):
##      return np.dot(theta, feature_vector) + theta_0
##  def classify_vector(feature_vector, theta, theta_0):
##      return 2*np.heaviside(decision_function(feature_vector, theta, theta_0), 0)-1
##  #pragma: coderesponse end



def classify(feature_matrix, theta, theta_0):
    """
    A classification function that uses given parameters to classify a set of
    data points.

    Args:
        `feature_matrix` - numpy matrix describing the given data. Each row
            represents a single data point.
        `theta` - numpy array describing the linear classifier.
        `theta_0` - real valued number representing the offset parameter.

    Returns:
        a numpy array of 1s and -1s where the kth element of the array is the
        predicted classification of the kth row of the feature matrix using the
        given theta and theta_0. If a prediction is GREATER THAN zero, it
        should be considered a positive classification.
    """
    epsilon = 10**(-8)
    prediction = feature_matrix @ theta + theta_0
    # 각 점이 어떤 값을 가질 수 있는지 판단 
    return np.where(prediction > epsilon, 1, -1)
    # 각 결과가 epsilon보다 크면 1, 작으면 -1을 리턴

    # Your code here
    raise NotImplementedError


def classifier_accuracy(
        classifier,
        train_feature_matrix,
        val_feature_matrix,
        train_labels,
        val_labels,
        **kwargs):
    """
    Trains a linear classifier and computes accuracy.  The classifier is
    trained on the train data.  The classifier's accuracy on the train and
    validation data is then returned.

    Args:
        `classifier` - A learning function that takes arguments
            (feature matrix, labels, **kwargs) and returns (theta, theta_0)
        `train_feature_matrix` - A numpy matrix describing the training
            data. Each row represents a single data point.
        `val_feature_matrix` - A numpy matrix describing the validation
            data. Each row represents a single data point.
        `train_labels` - A numpy array where the kth element of the array
            is the correct classification of the kth row of the training
            feature matrix.
        `val_labels` - A numpy array where the kth element of the array
            is the correct classification of the kth row of the validation
            feature matrix.
        `kwargs` - Additional named arguments to pass to the classifier
            (e.g. T or L)

    Returns:
        a tuple in which the first element is the (scalar) accuracy of the
        trained classifier on the training data and the second element is the
        accuracy of the trained classifier on the validation data.
    """
    theta, theta_0 = classifier(train_feature_matrix, train_labels, **kwargs)

    training_preds = classify(train_feature_matrix, theta, theta_0)
    validation_preds = classify(val_feature_matrix, theta, theta_0)

    training_accu = accuracy(training_preds, train_labels)
    validation_accu = accuracy(validation_preds, val_labels)

    return (training_accu, validation_accu)

    raise NotImplementedError



def extract_words(input_string):
    """
    Helper function for `bag_of_words(...)`.
    Args:
        a string `text`.
    Returns:
        a list of lowercased words in the string, where punctuation and digits
        count as their own words.
    """
    for c in punctuation + digits:
        input_string = input_string.replace(c, ' ' + c + ' ')

    return input_string.lower().split()
    raise NotImplementedError

    for c in punctuation + digits:
        text = text.replace(c, ' ' + c + ' ')
    return text.lower().split()



def bag_of_words(texts):
    """
    NOTE: feel free to change this code as guided by Section 3 (e.g. remove
    stopwords, add bigrams etc.)

    Args:
        `texts` - a list of natural language strings.
    Returns:
        a dictionary that maps each word appearing in `texts` to a unique
        integer `index`.
    """
    # Your code here
    f = open('stopwords.txt')
    stopwords = extract_words(f.read())
    f.close()

    dictionary = {} # maps word to unique index
    for text in texts:
        word_list = extract_words(text)
        for word in word_list:
            if word not in dictionary and word not in stopwords:
                dictionary[word] = len(dictionary)
    return dictionary
    raise NotImplementedError
    
    indices_by_word = {}  # maps word to unique index
    for text in texts:
        word_list = extract_words(text)
        for word in word_list:
            if word in indices_by_word: continue
            if word in stopword: continue
            indices_by_word[word] = len(indices_by_word)

    return indices_by_word

def extract_bow_feature_vectors(reviews, dictionary):
    """
    Inputs a list of string reviews
    Inputs the dictionary of words as given by bag_of_words
    Returns the bag-of-words feature matrix representation of the data.
    The returned matrix is of shape (n, m), where n is the number of reviews
    and m the total number of entries in the dictionary.

    Feel free to change this code as guided by Problem 9
    """
    # Your code here

    num_reviews = len(reviews)
    feature_matrix = np.zeros([num_reviews, len(dictionary)])

    for i, text in enumerate(reviews):
        word_list = extract_words(text)
        for word in word_list:
            if word in dictionary:
                feature_matrix[i, dictionary[word]] = 1
    return feature_matrix


# def extract_bow_feature_vectors(reviews, indices_by_word, binarize=True):
#     """
#     Args:
#         `reviews` - a list of natural language strings
#         `indices_by_word` - a dictionary of uniquely-indexed words.
#     Returns:
#         a matrix representing each review via bag-of-words features.  This
#         matrix thus has shape (n, m), where n counts reviews and m counts words
#         in the dictionary.
#     """
#     # Your code here
#     feature_matrix = np.zeros([len(reviews), len(indices_by_word)], dtype=np.float64)
#     for i, text in enumerate(reviews):
#         word_list = extract_words(text)
#         for word in word_list:
#             if word not in indices_by_word: continue
#             feature_matrix[i, indices_by_word[word]] += 1
#     if binarize:
#         # Your code here
#         raise NotImplementedError
#     return feature_matrix



def accuracy(preds, targets):
    """
    Given length-N vectors containing predicted and target labels,
    returns the fraction of predictions that are correct.
    """
    return (preds == targets).mean()
