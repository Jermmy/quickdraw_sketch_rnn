from sketch_rnn.model import SketchRNN
from sketch_rnn.utils import SketchLoader, get_sketch_labels
from sketch_rnn.config import model_file, log_dir
import tensorflow as tf


saving_model = True
loading_model = False

def accuracy(prediction, label):
    correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(label, 1))
    return tf.reduce_mean(tf.cast(correct_pred, tf.float32))


if __name__ == '__main__':
    batch_size = 100
    epoch = 2
    dictionary, reverse_dict = get_sketch_labels()
    n_class = len(dictionary)

    sl = SketchLoader(batch_size=batch_size, epoch=epoch)

    iterator = sl.train_dataset.make_one_shot_iterator()

    sketch, label, sketch_len = iterator.get_next()

    sketchrnn = SketchRNN(sketch, label, sketch_len,
                          n_class=len(dictionary), cell_hidden=[128, 256], n_hidden=128)

    batch_acc = accuracy(tf.nn.softmax(sketchrnn.pred), tf.one_hot(label, n_class))

    optimizer = tf.train.RMSPropOptimizer(learning_rate=0.01).minimize(sketchrnn.cost)

    # tf.summary.scalar("loss", sketchrnn.cost)
    # tf.summary.scalar("batch_acc", batch_acc)
    # # tf.summary.scalar("valid_acc", valid_acc)
    # merged_summary_op = tf.summary.merge_all()

    with tf.Session() as sess:
        saver = tf.train.Saver()
        # summary_writer = tf.summary.FileWriter(log_dir, graph=tf.get_default_graph())

        if loading_model:
            saver.restore(sess, model_file)
            print("model restored")
        else:
            tf.global_variables_initializer().run()

        try:
            step = 0
            while True:
                _, l, acc = sess.run([optimizer, sketchrnn.cost, batch_acc])

                # summary_writer.add_summary(summary, step)

                if step % 10 == 0:
                    print("Minibatch at step %d ===== loss: %.2f, acc: %.2f" % (step, l, acc))

                    try:
                        valid_iterator = sl.valid_dataset.make_one_shot_iterator()
                        valid_sketch, valid_label, valid_sketch_len = valid_iterator.get_next()
                        valid_pred = sketchrnn.network(valid_sketch, valid_sketch_len, reuse=True)
                        valid_acc = accuracy(tf.nn.softmax(valid_pred), tf.one_hot(valid_label, n_class))
                        [acc] = sess.run([valid_acc])
                        # summary_writer.add_summary(summary, step)
                        print("validation dataset ===== acc: %.2f" % (acc))
                    except tf.errors.OutOfRangeError:
                        pass

                if (step + 1) % 1000 == 0 and saving_model:
                    save_path = saver.save(sess, model_file)
                    print("Model saved in %s" % save_path)

                step += 1

        except tf.errors.OutOfRangeError:

            test_iterator = sl.test_dataset.make_one_shot_iterator()
            test_sketch, test_label, test_sketch_len = test_iterator.get_next()
            test_pred = sketchrnn.network(test_sketch, test_sketch_len, reuse=True)
            test_acc = accuracy(tf.nn.softmax(test_pred), tf.one_hot(test_label, n_class))
            try:
                [acc] = sess.run([test_acc])
                print("test dataset ===== acc: %.2f" % (acc))
            except tf.errors.OutOfRangeError:
                pass

            if saving_model:
                save_path = saver.save(sess, model_file)
                print("Model saved in %s" % save_path)
