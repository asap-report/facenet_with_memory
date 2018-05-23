import numpy as np
np.random.seed(123)

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from sklearn.metrics import confusion_matrix

from collections import OrderedDict
from pathlib import Path

import tensorflow as tf
tf.set_random_seed(123)

from facenet.src import facenet
from memory import Memory


# For FaceNet
IMAGE_SIZE = 160
MODEL_ID = '20170512-110547'
# MODEL_ID = '20180402-114759'
# MODEL_ID = '20180408-102900'
#
# You can find the above models at these URLs:
# 20170512-110547:
#    https://drive.google.com/file/d/0B5MzpY9kBtDVZ2RpVDYwWmxoSUk/edit
# 20180402-114759:
#    https://drive.google.com/open?id=1R77HmFADxe87GmoLwzfgMu_HY0IhcyBz
# 20180408-102900:
#    https://drive.google.com/open?id=1EXPBSXwTaqrSC0OhUdXNmKSh9qJUQ55-

# For reading images
DATASET = Path('./synthetic_faces/')
CHARACTER_ORDER = (
    'geralt',
    'vesemir',

    'malczewski',
    'witkacy',

    'avatar_female',
    'avatar_male',

    'gollum',
    'smeagol',

    'durotan',
    'hulk',

    'thade',
    'cesar',
)

# For rendering plots
TITLE_SIZE = 25
TICKS_SIZE = TITLE_SIZE - 8
AX_LABEL_SIZE = TITLE_SIZE - 4
FIG_SIZE = (10, 10)
IMG_PATH = Path('img')


def read_images(do_prewhiten=True):
    """Reads in the facial images of the 12 characters."""
    char_img_map = OrderedDict()
    for character in CHARACTER_ORDER:
        path = DATASET / character
        image_paths = sorted(path.glob('*.jpg'))
        images = facenet.load_data(image_paths, False, False, IMAGE_SIZE, do_prewhiten)
        char_img_map[character] = images

    return char_img_map


def get_num_images_per_character(char_img_map):
    """Returns the number of images loaded into the dictionary `char_img_map`
    per character and asserts that each character has the same number of images.
    """
    num_images = np.unique([len(images) for images in char_img_map.values()])
    assert len(num_images) == 1
    return num_images[0]


def get_image_size(char_img_map):
    x_shape, y_shape, _ = char_img_map.values()[0].shape
    assert x_shape == y_shape
    return x_shape


def load_facenet(my_graph, my_sess):
    """Loads the FaceNet model into the Graph `my_graph` and the Session
    `my_sess`.
    """
    with my_graph.as_default():
        with my_sess.as_default():
            facenet.load_model(MODEL_ID)

    # We don't need the contexts to be able to extract tensors from the graph
    images_pholder = my_graph.get_tensor_by_name('input:0')
    phase_train_pholder = my_graph.get_tensor_by_name('phase_train:0')
    embeddings_var = my_graph.get_tensor_by_name('embeddings:0')

    return images_pholder, phase_train_pholder, embeddings_var


def calc_embeddings(char_img_map, my_sess, images_pholder, phase_train_pholder,
                    embeddings_var):
    """Uses the FaceNet model to calculate the embeddings for the images provided
    in the `char_img_map` dict.

    Args:
        char_img_map (OrderedDict): Dictionary of: character -> `np.ndarray` of
            images.
        my_sess (tf.Session): Session in which the embeddings will be evaluated.
        images_pholder (tf.Tensor): Placeholder for input images.
        phase_train_pholder (tf.Tensor): Placeholder for the train phase (`True`
            or `False`).
        embeddings_var (tf.Tensor): Tensor corresponding to the embeddings variable
            in the FaceNet architecure.

    Returns:
        OrderedDict: Dictionary of: character -> `np.ndarray` of embeddings.
    """
    char_embedd_map = OrderedDict()
    for character in char_img_map.keys():
        char_embedd_map[character] = my_sess.run(embeddings_var, {
            images_pholder: char_img_map[character],
            phase_train_pholder: False,
        })

    return char_embedd_map


def plot_sim_mat(char_embedd_map):
    """Plot the matrix of cosine similarities between the embeddings.
    Actually, we're just calculating the dot product between the embeddings,
    because we're assuming they're normalized (and then the dot product is
    the same as the cosine similarity).
    """
    embeddings = np.concatenate([
        embedd for embedd in char_embedd_map.values()
    ])

    cossims_mat = embeddings.dot(embeddings.T)

    num_images_per_character = get_num_images_per_character(char_embedd_map)
    tick_marks = range(num_images_per_character//2, len(embeddings),
                       num_images_per_character)
    character_names = char_embedd_map.keys()

    plt.figure(figsize=FIG_SIZE)
    plt.imshow(cossims_mat)

    plt.xticks(tick_marks, character_names, rotation=90, fontsize=TICKS_SIZE)
    plt.yticks(tick_marks, character_names, fontsize=TICKS_SIZE)
    plt.grid(False)

    plt.savefig(str(IMG_PATH / 'sim_mat.png'))


def prepare_batch(char_entity_map, entity_idx, train_phase=True):
    """Creates a batch of "entities" (images of embeddings) that are then
    used to train or to evaluate a model in a k-shot manner.
    """
    num_images_per_character = get_num_images_per_character(char_entity_map)
    indexes = [entity_idx] if train_phase else range(entity_idx, num_images_per_character)

    batch_X = []
    batch_y = []
    for idx in indexes:
        for character_id, character in enumerate(char_entity_map.keys()):
            batch_X.append(char_entity_map[character][idx])
            batch_y.append(character_id)

    return np.r_[batch_X], np.array(batch_y, dtype=np.int32)


def nearest_neighbor_predictions(query_embedds, char_embedd_map, boundary_idx):
    """Find labels corresponding to observations in `query_embeddings`.
    The idea is that we can use observations in `char_embedd_map` but only up to
    index `boundary_idx` (inclusive). Observations with higher indexes are used
    for evaluation
    """
    num_characters = len(char_embedd_map)

    # Without any knowledge we migh as well throw a dice to make a prediction
    if boundary_idx == 0:
        return np.random.randint(0, num_characters, size=len(query_embedds))

    # We're looking for the embedding that's closest to the `query`, and use its
    # label as our prediction for this `query` (1-nearest-neighbor, essentially)
    predictions = []
    for query in query_embedds:
        closest_cossims = [
            char_embedd_map[char][:boundary_idx].dot(query).max()
            for char in char_embedd_map.keys()
        ]
        predictions.append(np.argmax(closest_cossims))

    return np.array(predictions)


def eval_facenet_without_memory(char_embedd_map):
    """Calculate predictions of the so-called "FaceNet alone" model (see the
    blog post for explanation).
    This function returns a dict `results` which is then used to calculate the
    k-shot accuracies of the model.
    """
    num_images_per_character = get_num_images_per_character(char_embedd_map)

    results = {}
    for image_idx in range(num_images_per_character):
        # Prediction phase
        partial_results = {}
        batch_X, batch_y = prepare_batch(char_embedd_map, image_idx, train_phase=False)
        partial_results['true'] = batch_y
        predictions = nearest_neighbor_predictions(batch_X, char_embedd_map, image_idx)
        partial_results['pred'] = predictions
        results[image_idx] = partial_results

        # Note: there's no training phase


    return results

def setup_memory(my_graph, embeddings_var, learning_rate=1e-3, memory_size=2**5,
                 vocab_size=2**3):
    """Prepares the memory module to be used with the FaceNet model."""
    with my_graph.as_default():
        embedding_size = embeddings_var.get_shape()[1]

        labels_placeholder = tf.placeholder(tf.int32, shape=[None])

        memory = Memory(
            key_dim=embedding_size,
            memory_size=memory_size,
            vocab_size=vocab_size,
        )

        mem_var_init_op = tf.variables_initializer(var_list=[
            memory.mem_keys,
            memory.mem_vals,
            memory.mem_age,
            memory.recent_idx,
            memory.query_proj,
        ])

        closest_label_train, _, teacher_loss_train = memory.query(
            embeddings_var, labels_placeholder, use_recent_idx=False)

        train_op = (tf.train
                    .GradientDescentOptimizer(learning_rate)
                    .minimize(teacher_loss_train))

        closest_label_pred, _, _ = memory.query(
            embeddings_var, None, use_recent_idx=False)

    return mem_var_init_op, labels_placeholder, closest_label_pred, train_op


def train_and_eval_facenet_with_memory(images,
                                       my_sess,
                                       images_pholder, phase_train_pholder, labels_placeholder,
                                       mem_var_init_op, train_op, closest_label_pred):
    """Similar to the `eval_facenet_without_memory` function, but this time we
    train and evaluate the "FaceNet + memory" model.
    But similarly to the `eval_facenet_without_memory` function, we return a
    dict `results` which is then used to calculate k-shot accuracies.
    """
    num_images_per_character = get_num_images_per_character(images)
    with my_sess.as_default():
        # Initialize the memory variables
        my_sess.run(mem_var_init_op)

        results = {}
        for image_idx in range(num_images_per_character):
            # Prediction phase
            partial_results = {}
            batch_X, batch_y = prepare_batch(images, image_idx, train_phase=False)
            partial_results['true'] = batch_y
            predictions = my_sess.run(closest_label_pred, {
                images_pholder: batch_X,
                phase_train_pholder: False,
            })
            partial_results['pred'] = predictions
            results[image_idx] = partial_results

            # Training phase
            # NOTE: BatchNorm with `phase_train_pholder` set to `True` may be problematic
            batch_X, batch_y = prepare_batch(images, image_idx)
            my_sess.run(train_op, {
                images_pholder: batch_X,
                labels_placeholder: batch_y,
                phase_train_pholder: False,
            })

    return results


def plot_results(results_facenet_alone, results_facenet_memory):
    filename = str(IMG_PATH / 'accuracies.png')
    _plot_accuracies(results_facenet_alone, results_facenet_memory, filename)
    plt.clf()

    title_prefix = 'FaceNet alone'
    filename = str(IMG_PATH / 'facenet_alone.gif')
    _animate_conf_mats(results_facenet_alone, title_prefix, filename)

    title_prefix = 'FaceNet + memory'
    filename = str(IMG_PATH / 'facenet_memory.gif')
    _animate_conf_mats(results_facenet_memory, title_prefix, filename)


def _plot_accuracies(results_facenet_alone, results_facenet_memory, filename):
    accuracies_facenet_alone = _extract_accuracies(results_facenet_alone)
    accuracies_facenet_with_memory = _extract_accuracies(results_facenet_memory)

    plt.figure(figsize=FIG_SIZE)
    plt.tick_params(labelsize=TICKS_SIZE)
    common_acc_kwargs = {
        'alpha': 0.5,
        'marker': 'o',
        'markersize': 10,
        'linestyle': '--',
    }
    plt.plot(
        accuracies_facenet_alone,
        label='FaceNet alone',
        **common_acc_kwargs
    )
    plt.plot(
        accuracies_facenet_with_memory,
        label='FaceNet + memory',
        **common_acc_kwargs
    )

    plt.title('k-shot accuracies', fontsize=TITLE_SIZE)
    plt.xlabel('k', fontsize=AX_LABEL_SIZE)
    plt.ylabel('Accuracy [%]', fontsize=AX_LABEL_SIZE)
    plt.ylim([0, 105])
    plt.legend(fontsize=TICKS_SIZE)

    plt.savefig(filename)
    plt.show()


def _extract_accuracies(results):
    accuracies = []
    for shot, result in results.items():
        labels, preds = result['true'], result['pred']
        acc = 100*(labels == preds).mean()
        accuracies.append(acc)

    return accuracies


def _animate_conf_mats(results, title_prefix, filename):
    fig = plt.figure(figsize=FIG_SIZE)

    cm = _calc_conf_mat(results[0]['true'], results[0]['pred'])
    im = plt.imshow(cm, animated=True, vmin=0, vmax=1)

    classes = CHARACTER_ORDER
    num_classes = len(classes)
    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, classes, rotation=90, fontsize=TICKS_SIZE)
    plt.yticks(tick_marks, classes, fontsize=TICKS_SIZE)
    plt.grid(False)

    # Needed to save the whole image, without cutting off the lower part
    fig.subplots_adjust(bottom=0.2)

    def animate(shot):
        true = results[shot]['true']
        pred = results[shot]['pred']
        acc = 100*(true==pred).mean()
        cm = _calc_conf_mat(true, pred)
        title = '{}, shot: {}\nAccuracy: {:.2f}%'.format(title_prefix, shot, acc)
        im.set_data(cm)
        plt.title(title, fontsize=TITLE_SIZE)
        return im,

    ax = plt.gca();
    ax.set_xticks(np.arange(0.5, num_classes), minor=True);
    ax.set_yticks(np.arange(0.5, num_classes), minor=True);
    ax.grid(which='minor', color='gray', linestyle='--', linewidth=2)

    ani = animation.FuncAnimation(fig, animate, range(len(results)), interval=3000, blit=True)
    ani.save(filename, dpi=100, writer='imagemagick')


def _calc_conf_mat(true, pred):
    cm = confusion_matrix(true, pred)
    cm = cm / cm.sum(axis=1).astype(np.float32)
    return cm
