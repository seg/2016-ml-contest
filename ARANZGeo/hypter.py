import pandas as pd

import classification_utilities
from networks_setups import *
from sklearn.metrics import confusion_matrix

#filename = 'training_data.csv'
testname = 'facies_vectors.csv'

facies_labels = ['SS', 'CSiS', 'FSiS', 'SiSh', 'MS', 'WS', 'D', 'PS', 'BS']
adjacent_facies = np.array([[1], [0, 2], [1], [4], [3, 5], [4, 6, 7], [5, 7], [5, 6, 8], [6, 7]])

TRAIN_RATIO = 0.35  # using the rule of thumb of 1/sqrt(num_input_variables)


def train_iteration(base_data, rand_index, sess, seed=1, print_epoch=False, iterations=200):
    # Split train/test set
    sample_index = -1 * int(TRAIN_RATIO * float(base_data.shape[0]))
    test_indices = rand_index[sample_index:]
    train_indices = rand_index[:sample_index]

    vals = base_data['Facies'].values[test_indices]
    vals -= 1
    hot_vals = np.zeros((vals.size, vals.max() + 1))
    hot_vals[np.arange(vals.size), vals] = 1

    test_labels_T = tf.convert_to_tensor(hot_vals)

    labels_w_noise, base_data = add_input_noise_from_facies(base_data, adjacent_facies, train_indices, seed=seed)
    labels_T = tf.convert_to_tensor(labels_w_noise)

    all_data = cleanup_csv(base_data)

    # Output data one hot between 1-9. Facies
    y_ = tf.placeholder(tf.float32, shape=[None, NUM_FACIES])

    # network setup
    y, x, features_T, test_features_T = three_layer_network(all_data.values[train_indices],
                                                            all_data.values[test_indices],
                                                            seed=seed, dropout=True)

    # loss function used to train
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))

    # backprop
    train_step = tf.train.AdamOptimizer(1e-3).minimize(cross_entropy)

    # Accuracy calculations
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # session init
    sess.run(tf.global_variables_initializer())

    for i in range(iterations):
        x_vals, y_labels, x_vals_t, y_labels_t = sess.run([features_T, labels_T, test_features_T, test_labels_T])

        train_data = {x: x_vals, y_: y_labels}
        _, train_acc = sess.run([train_step, accuracy], feed_dict=train_data)

        test_data = {x: x_vals_t, y_: y_labels_t}
        test_acc = sess.run(accuracy, feed_dict=test_data)

        if i % 1000 == 0 and print_epoch:
            print('epoch', i / 1000)
            print('test acc', test_acc)
            print('train acc', train_acc, '\n')

    print('test acc', test_acc)
    print('train acc', train_acc)

    real, predicted = y_labels, sess.run(y, feed_dict=train_data)
    real_test, predicted_test = y_labels_t, sess.run(y, feed_dict=test_data)

    real = np.argmax(real, axis=1)

    real_t = np.argmax(real_test, axis=1)
    predicted_t1 = np.argmax(predicted_test, axis=1)
    conf2 = confusion_matrix(real_t, predicted_t1)

    validation = 'validation_data_nofacies.csv'
    validation_data = pd.read_csv(validation)
    v_data = cleanup_csv(validation_data)

    final_t = sess.run(y, feed_dict={x: v_data.values})

    precision = np.diagonal(conf2)/conf2.sum(axis=0).astype('float')
    recall = np.diagonal(conf2)/conf2.sum(axis=1).astype('float')
    F1 = 2 * (precision * recall) / (precision + recall)
    F1[np.isnan(F1)] = 0

    return final_t, predicted_test, real_t, F1, predicted, real


# We are using a weighted majority stacking
seeds = [10, 4, 12]
sess = tf.Session()

base_data = pd.read_csv(testname)

validation = 'validation_data_nofacies.csv'
validation_data = pd.read_csv(validation)

test_prior = None
final_prior = None
final_predictions = None

for seed in seeds:
    np.random.seed(seed)
    rand_index = np.random.permutation(np.arange(base_data.shape[0]))
    final_t, test_t, real_test_labels, F1, p_t, real_label = train_iteration(base_data, rand_index, sess, seed, True, 20000)

    facies = np.argmax(test_t, axis=1)
    facies_r = np.argmax(p_t, axis=1)
    final_facies = np.argmax(final_t, axis=1)

    # Majority voting
    hot_vals = np.zeros((facies.size, NUM_FACIES))
    hot_vals[np.arange(facies.size), facies] = 1

    hot_vals_2 = np.zeros((final_facies.size, NUM_FACIES))
    hot_vals_2[np.arange(final_facies.size), final_facies] = 1

    conf2 = confusion_matrix(np.argmax(hot_vals, axis=1), real_test_labels)
    classification_utilities.display_cm(conf2, facies_labels, display_metrics=True, hide_zeros=True)

    final_weights = hot_vals_2 * F1

    if final_prior is None:
        final_prior = final_weights
    else:
        final_prior = np.add(final_weights, final_prior)

    if final_predictions is not None:
        print('\ndiff:')
        print(np.sum(final_predictions != (np.argmax(final_weights, axis=1) + 1)) / len(final_predictions), '\n')
        print(np.sum(final_predictions != (np.argmax(final_prior, axis=1) + 1)) / len(final_predictions), '\n')

    # # Get the final predictions and index back to 1-9
    final_predictions = np.argmax(final_prior, axis=1) + 1

validation_data['Facies'] = final_predictions
validation_data.to_csv('ARANZGeo/final_predictions_cross.csv')