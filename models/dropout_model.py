from models.model import Model
from keras import models, layers
from keras.optimizers import Adam

class DropoutModel(Model):
    def _define_model(self, input_shape, categories_count):
        # Your code goes here

        network = models.Sequential()


        network.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=input_shape))
        network.add(layers.MaxPooling2D((2,2)))

        network.add(layers.Conv2D(64, (3,3), activation='relu', input_shape=input_shape))
        network.add(layers.MaxPooling2D((2,2)))
        network.add(layers.Dropout(0.15))

        network.add(layers.Conv2D(128, (3,3), activation='relu', input_shape=input_shape))
        network.add(layers.MaxPooling2D((2,2)))
        network.add(layers.Dropout(0.35))

        network.add(layers.Flatten())

        

        network.add(layers.Dense(128, activation='relu'))
        network.add(layers.Dense(categories_count, activation='softmax'))

        self.model = network

        
    
    def _compile_model(self):
        # Your code goes here

        # self.model.compile(<configuration properties>)
        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy'],
        )