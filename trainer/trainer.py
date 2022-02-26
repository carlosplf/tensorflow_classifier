import os
import tensorflow as tf


class ConvTrainer():
    def __init__(self, model, train_ds, valid_ds):
        self.model = model
        self.train_ds = train_ds
        self.valid_ds = valid_ds

    def do_training_cycle(self, epochs):
        
        checkpoint_path = "checkpoint/cp.ckpt"
        checkpoint_dir = os.path.dirname(checkpoint_path)
        
        # Create a callback that saves the model's weights
        cp_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            save_weights_only=True,
            verbose=1
        )

        history = self.model.fit(
            self.train_ds,
            validation_data=self.valid_ds,
            epochs=epochs,
            callbacks=[cp_callback]
        )
        
        return history
