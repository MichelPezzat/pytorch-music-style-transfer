import tensorflow as tf


class Logger(object):
    """Tensorboard logger."""

    def __init__(self, log_dir):
        """Initialize summary writer."""
        self.writer = tf.summary.create_file_writer(log_dir)
        

    def scalar_summary(self, tag, value, step):
        """Add scalar summary."""
        summary = tf.summary.scalar(tag,value,step=step)
        self.writer.flush()