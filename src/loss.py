import tensorflow as tf
from tensorflow.python.keras.losses import Loss


class Seq2SeqLoss(Loss):
    def __init__(self, target_mask, batch_size):
        super(Seq2SeqLoss, self).__init__()
        self.target_mask = target_mask
        self.batch_size = batch_size

    def call(self, y_true, y_pred):
        loss_per_step = []

        batch_num = tf.range(0, self.batch_size)

        for dec_step, dist in y_pred:
            target = y_true[:, dec_step]
            indices = tf.stack([batch_num, target])
            gold_probs = tf.gather_nd(dist, indices)
            loss_ = - tf.log(gold_probs)
            loss_per_step.append(loss_)

        # mask
        dec_len = tf.reduce_sum(self.target_mask)
        values_per_step = [v * self.target_mask[:, dec_step] for dec_step, v in enumerate(loss_per_step)]
        values_per_ex = tf.reduce_sum(values_per_step) / dec_len
        return tf.reduce_mean(values_per_ex)