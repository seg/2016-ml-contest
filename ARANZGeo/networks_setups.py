import tensorflow as tf
import numpy as np

NUM_FACIES = 9
NUM_INPUTS = 9

forms_replace = {'A1 LM': 1,
                 'A1 SH': 2,
                 'B1 LM': 3,
                 'B1 SH': 4,
                 'B2 LM': 5,
                 'B2 SH': 6,
                 'B3 LM': 7,
                 'B3 SH': 7,
                 'B4 LM': 8,
                 'B4 SH': 9,
                 'B5 LM': 10,
                 'B5 SH': 11,
                 'C LM': 12,
                 'C SH': 13}


def add_input_noise_from_facies(data, adjacent_facies, train_indices, noise_pecentage=0.05, seed=1):
    noise_dict = {i: k for i, k in enumerate(adjacent_facies)}

    vals = data['Facies'].values[train_indices]
    vals -= 1

    np.random.seed(seed)
    random_index = np.random.permutation(np.arange(int(len(vals) * noise_pecentage)))
    to_randomise = vals[random_index]
    randomed = [np.random.choice(noise_dict[arr]) for arr in to_randomise]
    vals[random_index] = randomed

    hot_vals = np.zeros((vals.size, vals.max() + 1))
    hot_vals[np.arange(vals.size), vals] = 1
    data = data.drop('Facies', axis=1)
    return hot_vals, data


def cleanup_csv(data, standardize=True):
    # ** Facies Formation	Well Name	Depth	GR	ILD_log10	DeltaPHI	PHIND	PE	NM_M    RELPOS
    # The well name is not relevant
    data = data.drop('Well Name', axis=1)

    # Replace formation and well names with corresponding integers
    data['Formation'] = data['Formation'].replace(forms_replace)

    # Fill missing values and normalize
    data['PE'] = data['PE'].fillna(value=0)
    for name in data:
        if standardize:
            data[name] = (data[name] - data[name].mean()) / data[name].std()
        else:
            data[name] = (data[name] - data[name].min()) / (data[name].max() - data[name].min())
    return data


def two_layer_network(all_data, sample_index, dropout=False):
    # setup
    l_x = tf.placeholder(tf.float32, shape=[None, NUM_INPUTS])
    features_T = tf.pack(all_data.values[:sample_index])
    test_features_T = tf.pack(all_data.values[sample_index:])

    # layer sizes
    k = 25
    l = 15

    # Weights and biases
    w1 = tf.Variable(tf.truncated_normal([NUM_INPUTS, k]))
    w2 = tf.Variable(tf.truncated_normal([k, l]))
    w3 = tf.Variable(tf.truncated_normal([l, NUM_FACIES]))

    b1 = tf.Variable(tf.zeros([k]))
    b2 = tf.Variable(tf.zeros([l]))
    b3 = tf.Variable(tf.zeros([NUM_FACIES]))

    # Regression matrix
    if dropout:
        y1d = tf.nn.relu(tf.matmul(l_x, w1) + b1)
        y1 = tf.nn.dropout(y1d, 0.9)
        y2d = tf.nn.relu(tf.matmul(y1, w2) + b2)
        y2 = tf.nn.dropout(y2d, 0.95)
    else:
        y1 = tf.nn.relu(tf.matmul(l_x, w1) + b1)
        y2 = tf.nn.relu(tf.matmul(y1, w2) + b2)

    l_y = tf.matmul(y2, w3) + b3
    return l_y, l_x, features_T, test_features_T


def three_layer_network(train_data, test_data, seed=None, dropout=False):
    # setup
    l_x = tf.placeholder(tf.float32, shape=[None, NUM_INPUTS])
    features_T = tf.pack(train_data)
    test_features_T = tf.pack(test_data)

    # layer sizes
    k = 100
    l = 55
    j = 20

    # Weights and biases
    w1 = tf.Variable(tf.truncated_normal([NUM_INPUTS, k], seed=seed))
    w2 = tf.Variable(tf.truncated_normal([k, l], seed=seed))
    w3 = tf.Variable(tf.truncated_normal([l, j], seed=seed))
    w4 = tf.Variable(tf.truncated_normal([j, NUM_FACIES], seed=seed))

    b1 = tf.Variable(tf.zeros([k]))
    b2 = tf.Variable(tf.zeros([l]))
    b3 = tf.Variable(tf.zeros([j]))
    b4 = tf.Variable(tf.zeros([NUM_FACIES]))

    # Regression matrix
    if dropout:
        y1d = tf.nn.relu(tf.matmul(l_x, w1) + b1)
        y1 = tf.nn.dropout(y1d, 0.9)
        y2d = tf.nn.relu(tf.matmul(y1, w2) + b2)
        # y2 = tf.nn.dropout(y2d, 0.95)
        y3d = tf.nn.relu(tf.matmul(y2d, w3) + b3)
        y3 = tf.nn.dropout(y3d, 0.95)
    else:
        y1 = tf.nn.relu(tf.matmul(l_x, w1) + b1)
        y2 = tf.nn.relu(tf.matmul(y1, w2) + b2)
        y3 = tf.nn.relu(tf.matmul(y2, w3) + b3)

    l_y = tf.matmul(y3, w4) + b4
    return l_y, l_x, features_T, test_features_T


def convolutional_network(all_data, sample_index):
    # setup
    c_x = tf.placeholder(tf.float32, shape=[None, NUM_INPUTS, 1])

    f = all_data.values[:sample_index]
    f = f[:, :, np.newaxis]

    t = all_data.values[sample_index:]
    t = t[:, :, np.newaxis]

    features_T = tf.pack(f)
    test_features_T = tf.pack(t)

    k = 3
    m = 6
    n = 25

    # of the form [filter-size, input channels, output channels]

    w1 = tf.Variable(tf.truncated_normal([4, 1, k]))  # stride 1
    w2 = tf.Variable(tf.truncated_normal([3, k, m]))  # stride 2
    w3 = tf.Variable(tf.truncated_normal([5 * m, n]))
    w4 = tf.Variable(tf.truncated_normal([n, NUM_FACIES]))

    b1 = tf.Variable(tf.ones([k]))
    b2 = tf.Variable(tf.ones([m]))
    b3 = tf.Variable(tf.ones([n]))
    b4 = tf.Variable(tf.ones([NUM_FACIES]))

    y1 = tf.nn.relu(tf.nn.conv1d(c_x, w1, stride=1, padding='SAME') + b1)
    y2 = tf.nn.relu(tf.nn.conv1d(y1, w2, stride=2, padding='SAME') + b2)

    yy = tf.reshape(y2, shape=[-1, 5 * m])
    y3 = tf.nn.relu(tf.matmul(yy, w3) + b3)

    c_y = tf.matmul(y3, w4) + b4

    return c_y, c_x, features_T, test_features_T