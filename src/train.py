from model import *
epochs = 2
batch_size = 128
history = model.fit(x_train, y_train, epochs=1, batch_size= batch_size,validation_split=0.1)
model.save('AttentionX.h5')