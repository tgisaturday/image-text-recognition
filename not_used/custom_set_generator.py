#reference: https://gist.github.com/ambodi/408301bc5bc07bc5afa8748513ab9477

"""A generic module to read data."""
import numpy
import collections
from tensorflow.python.framework import dtypes


class DataSet(object):
    """Dataset class object."""

    def __init__(self, images, labels, fake_data=False, one_hot=False, dtype=dtypes.float64,reshape=True):
        """Initialize the class."""
        if reshape:
            assert images.shape[3] == 1
            images = images.reshape(images.shape[0], images.shape[1] * images.shape[2])

        self._images = images
        self._num_examples = images.shape[0]
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size, fake_data=False):
        """Return the next `batch_size` examples from this data set."""
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            perm = numpy.arange(self._num_examples)
            numpy.random.shuffle(perm)
            self._images = self._images[perm]
            self._labels = self._labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch

        return self._images[start:end], self._labels[start:end]


def read_data_sets(train_dir, fake_data=False, one_hot=False, dtype=dtypes.float64, reshape=True,validation_size=5000):
    """Set the images and labels."""
    num_training = 3000 # number of training sets
    num_validation = 1000
    num_test = 1000

    all_images = numpy.load('./custom_data/custom_image.npy')
    all_images = all_images.reshape(all_images.shape[0], all_images.shape[1], all_images.shape[2], 1)

    train_labels_original = numpy.load('./custom_data/custom_label.npy')
    all_labels = numpy.asarray(range(0, len(train_labels_original)))
    all_labels = dense_to_one_hot(all_labels, len(all_labels))

    mask = range(num_training)
    train_images = all_images[mask]
    train_labels = all_labels[mask]

    mask = range(num_training, num_training + num_validation)
    validation_images = all_images[mask]
    validation_labels = all_labels[mask]

    mask = range(num_training + num_validation, num_training + num_validation + num_test)
    test_images = all_images[mask]
    test_labels = all_labels[mask]

    train = DataSet(train_images, train_labels, dtype=dtype, reshape=reshape)
    validation = DataSet(validation_images, validation_labels, dtype=dtype,
        reshape=reshape)

    test = DataSet(test_images, test_labels, dtype=dtype, reshape=reshape)
    ds = collections.namedtuple('Datasets', ['train', 'validation', 'test'])

    return ds(train=train, validation=validation, test=test)



def dense_to_one_hot(labels_dense, num_classes):
    """Convert class labels from scalars to one-hot vectors."""
    num_labels = labels_dense.shape[0]
    index_offset = numpy.arange(num_labels) * num_classes
    labels_one_hot = numpy.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1

    return labels_one_hot