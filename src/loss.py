import tensorflow as tf
from tensorflow.python.keras.losses import Loss
from tensorflow.python.keras.layers import Lambda

def losses(decode_len, target_mask, batch_size, vocab_dist, target_id):
    loss_per_step = []

    batch_num = tf.range(0, batch_size, dtype=tf.int32)
    # y_pred = Lambda(lambda x: tf.unstack(x))(vocab_dist)
    for dec_step in range(decode_len):
        dist = vocab_dist[:,dec_step,:]
        target = target_id[:, dec_step]
        indices = tf.stack([batch_num, target],axis=1)
        gold_probs = tf.gather_nd(dist, indices)
        loss_ = - tf.log(gold_probs)
        loss_per_step.append(loss_)
    # mask
    dec_len = tf.reduce_sum(target_mask)
    values_per_step = [v * tf.cast(target_mask[:, dec_step],dtype=tf.float32) for dec_step, v in enumerate(loss_per_step)]
    values_per_ex = tf.reduce_sum(values_per_step) / tf.cast(dec_len,tf.float32)
    return tf.reduce_mean(values_per_ex)


class Seq2SeqLoss(Loss):
    def __init__(self, target_mask, batch_size):
        super(Seq2SeqLoss, self).__init__()
        self.target_mask = target_mask
        self.batch_size = batch_size

    def call(self, y_true, y_pred):
        loss_per_step = []
        y_true = tf.cast(y_pred,dtype=tf.int32)

        batch_num = tf.range(0, self.batch_size, dtype=tf.int32)
        y_pred = Lambda(lambda x: tf.unstack(x))(y_pred)
        for dec_step in range(10):
            dist = y_pred[dec_step]
            target = y_true[:, dec_step]
            indices = tf.stack([batch_num, target])
            gold_probs = tf.gather_nd(dist, indices)
            loss_ = - tf.log(gold_probs)
            loss_per_step.append(loss_)

        # mask
        dec_len = tf.reduce_sum(self.target_mask)
        values_per_step = [v * self.target_mask[:, dec_step] for dec_step, v in enumerate(loss_per_step)]
        values_per_ex = tf.reduce_sum(values_per_step) / tf.cast(dec_len,dtype=tf.float32)
        return tf.reduce_mean(values_per_ex)