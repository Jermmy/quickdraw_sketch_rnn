from sketch_rnn.model import online_model
from sketch_rnn.utils import SketchLoader, get_sketch_labels
from sketch_rnn.config import model_file, log_dir
import tensorflow as tf
import numpy as np


saving_model = True
loading_model = False


def accuracy(prediction, label):
    correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(label, 1))
    return tf.reduce_mean(tf.cast(correct_pred, tf.float32))


if __name__ == '__main__':
    batch_size = 128
    epoch = 5
    dictionary, reverse_dict = get_sketch_labels()
    n_class = len(dictionary)

    sl = SketchLoader(batch_size=batch_size, epoch=epoch)

    iterator = sl.train_dataset.make_one_shot_iterator()

    sketch, label, sketch_len = iterator.get_next()

    sketchrnn = online_model(n_class=n_class, cell_hidden=[256, 512])
    pred, cost = sketchrnn.train(sketch, label, sketch_len)

    batch_acc = accuracy(tf.nn.softmax(pred), tf.one_hot(label, n_class))

    global_step = tf.Variable(0, trainable=False)
    starter_learning_rate = 0.001
    learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, 2000, 0.7, staircase=True)
    # optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(cost, global_step=global_step)
    optimizer = tf.contrib.layers.optimize_loss(
        loss=cost,
        global_step=global_step,
        learning_rate=learning_rate,
        optimizer=tf.train.RMSPropOptimizer,
        clip_gradients=9.0,
        summaries=["learning_rate", "loss"]
    )

    tf.summary.scalar("loss", cost)
    tf.summary.scalar("batch_acc", batch_acc)
    # # tf.summary.scalar("valid_acc", valid_acc)
    merged_summary_op = tf.summary.merge_all()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        saver = tf.train.Saver()
        summary_writer = tf.summary.FileWriter(log_dir, graph=tf.get_default_graph())

        if loading_model:
            saver.restore(sess, model_file)
            print("model restored")
        else:
            tf.global_variables_initializer().run()

        try:
            step = 0
            while True:
                _, l, acc, summary = sess.run([optimizer, cost, batch_acc, merged_summary_op])

                summary_writer.add_summary(summary, step)

                if step % 500 == 0:
                    print("Minibatch at step %d ===== loss: %.2f, acc: %.2f" % (step, l, acc))

                if (step + 1) % 1000 == 0 and saving_model:
                    save_path = saver.save(sess, model_file)
                    print("Model saved in %s" % save_path)

                step += 1

        except tf.errors.OutOfRangeError:
            test_accs = []
            test_iterator = sl.test_dataset.make_one_shot_iterator()
            test_sketch, test_label, test_sketch_len = test_iterator.get_next()
            test_pred = sketchrnn.inference(test_sketch, test_sketch_len, reuse=True)
            test_acc = accuracy(test_pred, tf.one_hot(test_label, n_class))
            try:
                while True:
                    acc = sess.run(test_acc)
                    # print("test dataset ===== acc: %.2f" % (acc))
                    test_accs.append(acc)
            except tf.errors.OutOfRangeError:
                # pass
                print("test dataset ===== acc: %.2f" % (np.mean(test_accs)))

            if saving_model:
                save_path = saver.save(sess, model_file)
                print("Model saved in %s" % save_path)

            print("Run the command line:\n--> tensorboard --logdir=/tmp/tensorflow_logs \n"
                  "Then open http://0.0.0.0:6006/ into your web browser")
