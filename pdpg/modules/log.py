import tensorflow as tf

try:
    from StringIO import StringIO  # Python 2.7
except ImportError:
    from io import BytesIO         # Python 3.x


class TfLogger(object):

    def __init__(self, log_dir):
        """Create a summary writer logging to log_dir."""
        self.writer = tf.summary.create_file_writer(log_dir)

    def scalar_summary(self, tag, value, step):
        """Log a scalar variable."""

        with tf.device("/cpu:0"):
            with self.writer.as_default():
                tf.summary.scalar(tag, value, step=step)

    def image_summary(self, tag, images, step):
        """Log tensor of images"""

        with tf.device("/cpu:0"):
            with self.writer.as_default():
                tf.summary.image(tag, images, step=step)

    def histo_summary(self, tag, values, step, bins=1000):
        """Log a histogram of the tensor of values."""

        with tf.device("/cpu:0"):
            with self.writer.as_default():
                tf.summary.histogram(tag, tf.convert_to_tensor(values), step=step, buckets=bins)
