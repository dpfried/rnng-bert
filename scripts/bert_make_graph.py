assert __name__ == "__main__", "This script is not designed to be imported"
BERT_CODE_PATH = "~/dev/bert"
BERT_MODEL_DIR = "uncased_L-12_H-768_A-12"
BERT_OUTPUT_FILE = "bert_graph.pb"
# bert version used: https://github.com/google-research/bert/tree/f39e881b169b9d53bea03d2d341b31707a6c052b
# BERT_CODE_PATH should be the parent folder of the repo, so "import bert" works

import sys, os
sys.path.append(os.path.expanduser(BERT_CODE_PATH))

import tensorflow as tf
import bert
import bert.modeling, bert.optimization

#%%

sess = tf.InteractiveSession()

# %%

config = bert.modeling.BertConfig.from_json_file(os.path.join(BERT_MODEL_DIR, "bert_config.json"))

# %%

input_ids = tf.placeholder(shape=(None, None), dtype=tf.int32, name='input_ids')
word_end_mask = tf.placeholder(shape=(None, None), dtype=tf.int32, name='word_end_mask')

# %%

input_mask = (1 - tf.cumprod(1 - word_end_mask, axis=-1, reverse=True))
token_type_ids = tf.zeros_like(input_ids)

# %%

#XXX(nikita): have both a training mode and an eval mode
model = bert.modeling.BertModel(config=config, is_training=False,
    input_ids=input_ids, input_mask=input_mask, token_type_ids=token_type_ids)


def get_word_features():
    subword_features = model.get_sequence_output()
    word_features_packed = tf.gather(
        tf.reshape(subword_features, [-1, int(subword_features.shape[-1])]),
        tf.to_int32(tf.where(tf.reshape(word_end_mask, (-1,))))[:,0])

    # input_mask is over subwords, whereas valid_mask is over words
    sentence_lengths = tf.reduce_sum(word_end_mask, -1)
    valid_mask = (tf.range(tf.reduce_max(sentence_lengths))[None,:] < sentence_lengths[:, None])
    dim_padded = tf.shape(valid_mask)[:2]
    mask_flat = tf.reshape(valid_mask, (-1,))
    dim_flat = tf.shape(mask_flat)[:1]
    nonpad_ids = tf.to_int32(tf.where(mask_flat)[:,0])
    word_features_padded = tf.scatter_nd(
        indices=nonpad_ids[:, None],
        updates=word_features_packed,
        shape=tf.concat([dim_flat, tf.shape(word_features_packed)[1:]], axis=0)
        )
    word_features_padded = tf.reshape(
        word_features_padded,
        tf.concat([dim_padded, [-1]], axis=0)
        )
    return word_features_padded

word_features = tf.identity(get_word_features(), name='word_features')
word_features_grad = tf.placeholder(shape=(None, None, word_features.shape[-1]), dtype=word_features.dtype, name='word_features_grad')

# %%

saver = tf.train.Saver()

# %% verify that restoring a checkpoint works

saver.restore(sess, os.path.join(BERT_MODEL_DIR, "bert_model.ckpt"))

# %%

def create_optimizer(ys, grad_ys, init_lr=5e-4, num_warmup_steps=160):
    """Creates an optimizer training op."""
    global_step = tf.train.get_or_create_global_step()

    # TODO(nikita): make ops for setting the optimizer hyperparameters
    if False:
        with sess.graph.as_default() as g, g.name_scope(None):
            learning_rate_var = learning_rate = tf.get_variable(
                "learning_rate",
                shape=[],
                dtype=tf.float32,
                initializer=tf.constant_initializer(init_lr),
                trainable=False,
                collections=[tf.GraphKeys.GLOBAL_VARIABLES])
            warmup_steps_var = warmup_steps_int = tf.get_variable(
                "warmup_steps",
                shape=[],
                dtype=tf.int32,
                initializer=tf.constant_initializer(num_warmup_steps, dtype=tf.int32),
                trainable=False,
                collections=[tf.GraphKeys.GLOBAL_VARIABLES])
    else:
        learning_rate = tf.constant(value=init_lr, shape=[], dtype=tf.float32)
        warmup_steps_int = tf.constant(value=num_warmup_steps, shape=[], dtype=tf.int32)
        learning_rate_var = warmup_steps_var = None

    # Implements linear warmup. I.e., if global_step < num_warmup_steps, the
    # learning rate will be `global_step/num_warmup_steps * init_lr`.
    global_steps_int = tf.cast(global_step, tf.int32)
    warmup_steps_int = tf.constant(num_warmup_steps, dtype=tf.int32)

    global_steps_float = tf.cast(global_steps_int, tf.float32)
    warmup_steps_float = tf.cast(warmup_steps_int, tf.float32)

    warmup_percent_done = global_steps_float / warmup_steps_float
    warmup_learning_rate = init_lr * warmup_percent_done

    is_warmup = tf.cast(global_steps_int < warmup_steps_int, tf.float32)
    learning_rate = (
    (1.0 - is_warmup) * learning_rate + is_warmup * warmup_learning_rate)

    # It is recommended that you use this optimizer for fine tuning, since this
    # is how the model was trained (note that the Adam m/v variables are NOT
    # loaded from init_checkpoint.)
    optimizer = bert.optimization.AdamWeightDecayOptimizer(
        learning_rate=learning_rate,
        weight_decay_rate=0.01,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-6,
        exclude_from_weight_decay=["LayerNorm", "layer_norm", "bias"])

    tvars = tf.trainable_variables()
    grad_ys = [tf.check_numerics(node, "Placeholder {}".format(node.name)) for node in grad_ys]
    # grads = tf.gradients(loss, tvars)
    grads = tf.gradients(ys=ys, xs=tvars, grad_ys=grad_ys)

    # This is how the model was pre-trained.
    (grads, _) = tf.clip_by_global_norm(grads, clip_norm=1.0)

    train_op = optimizer.apply_gradients(
      zip(grads, tvars), global_step=global_step)

    # Normally the global step update is done inside of `apply_gradients`.
    # However, `AdamWeightDecayOptimizer` doesn't do this. But if you use
    # a different optimizer, you should probably take this line out.
    new_global_step = global_step + 1
    train_op = tf.group(train_op, [global_step.assign(new_global_step)], name='train')
    return train_op, learning_rate_var, warmup_steps_var

# %%

train_op, learning_rate, warmup_steps = create_optimizer([word_features], [word_features_grad])

# %%

init_op = tf.variables_initializer(tf.global_variables(), name='init')

# %%

print(f"""
Operation names:

input_ids: {input_ids.name}
word_end_mask: {word_end_mask.name}
word_features: {word_features.name}
word_features_grad: {word_features_grad.name}
init_op: {init_op.name}
train_op: {train_op.name}
save_op: {saver.saver_def.save_tensor_name}
restore_op: {saver.saver_def.restore_op_name}
checkpoint_name: {saver.saver_def.filename_tensor_name}
""")

# %%

with open(BERT_OUTPUT_FILE, 'wb') as f:
    f.write(sess.graph_def.SerializeToString())

print("Saved tensorflow graph to to", BERT_OUTPUT_FILE)

# %%
