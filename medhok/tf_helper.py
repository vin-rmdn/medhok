from . import constants as c
import tensorflow as tf


def serialize_example(feats, dialects):
    feature = {
        'feature': c.bytes_feature(tf.io.serialize_tensor(feats).numpy()),
        'dialect': c.bytes_feature(dialects)
    }

    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()


def tf_serialize_example(feats, dialects):
    tf_string = tf.py_function(
        serialize_example,
        (feats, dialects),
        tf.string
    )
    return tf.reshape(tf_string, ())
