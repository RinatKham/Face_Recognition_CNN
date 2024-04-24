import os
import random
import pytesseract
import matplotlib.pyplot as plt

train_loss = [7.139, 4.713, 3.911, 3.437, 3.244, 2.963, 2.776, 2.679, 2.623, 2.427]
avg_ac = [0.85606, 0.88636, 0.93939, 0.97727, 0.98485, 1, 0.96970, 0.98485, 1, 1]

plt.figure(figsize=(15, 5))

# Plotting the loss over epochs
plt.subplot(121)
plt.plot(train_loss, 'b', label='Loss')
plt.title('Training loss')
plt.legend()

# Plotting the accuracy over epochs
plt.subplot(122)
plt.plot(avg_ac, 'r', label='Accuracy')
plt.title('Testing Accuracy')
plt.legend()

plt.show()